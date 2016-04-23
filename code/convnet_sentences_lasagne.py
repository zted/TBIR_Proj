import numpy as np
import warnings
from conv_net_classes import *
import lasagne as L
import lasagne.layers as LL
import sys
from itertools import islice
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


def train_conv_net(datasets,
                   hidden_units,
                   num_filters=[32, 32, 32],
                   filter_hs=[3, 4, 5],
                   dropout_rate=[0.5],
                   n_epochs=25,
                   batch_size=50,
                   load_model=False):
    x_train, y_train, x_val, y_val = datasets

    assert len(num_filters) == len(filter_hs)
    img_w = len(x_train[0][0][0])
    img_h = len(x_train[0][0])
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((filter_h, img_w))
        pool_sizes.append((img_h - filter_h + 1, 1))

    input_shape = x_train[0].shape
    l_in = LL.InputLayer(
        shape=(None, input_shape[0], input_shape[1], input_shape[2]))

    layer_list = []
    for i in range(len(filter_hs)):
        l_conv = LL.Conv2DLayer(l_in, num_filters=num_filters[i], filter_size=filter_shapes[i],
                                nonlinearity=L.nonlinearities.rectify,
                                W=L.init.HeNormal(gain='relu'))
        l_pool = LL.MaxPool2DLayer(l_conv, pool_size=pool_sizes[i])
        layer_list.append(l_pool)

    mergedLayer = LL.ConcatLayer(layer_list)

    l_hidden1 = LL.DenseLayer(mergedLayer, num_units=hidden_units[0],
                              nonlinearity=L.nonlinearities.tanh,
                              W=L.init.HeNormal(gain='relu'))
    l_hidden1_dropout = LL.DropoutLayer(l_hidden1, p=dropout_rate[0])

    l_hidden2 = LL.DenseLayer(l_hidden1_dropout, num_units=hidden_units[1],
                              nonlinearity=L.nonlinearities.tanh,
                              W=L.init.HeNormal(gain='relu'))
    l_hidden2_dropout = LL.DropoutLayer(l_hidden2, p=dropout_rate[1])

    l_output = LL.DenseLayer(l_hidden2_dropout, num_units=hidden_units[2],
                             nonlinearity=L.nonlinearities.tanh)
    net_output_train = LL.get_output(l_output, deterministic=False)

    if (load_model):
        with np.load('../data/model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        LL.set_all_param_values(l_output, param_values)

    true_output = T.matrix('true_output', dtype='float32')

    loss_train = T.sum(L.objectives.squared_error(net_output_train, true_output)) / net_output_train.shape[0]
    all_params = LL.get_all_params(l_output)

    # Use ADADELTA for updates
    updates = L.updates.adadelta(loss_train, all_params)
    train = theano.function([l_in.input_var, true_output], loss_train, updates=updates)

    # This is the function we'll use to compute the network's output given an input
    # (e.g., for computing accuracy).  Again, we don't want to apply dropout here
    # so we set the deterministic kwarg to True.
    net_output_val = theano.function([l_in.input_var],
                                     LL.get_output(l_output, deterministic=True))

    # x_train, y_train = shared_dataset((x_train, y_train))
    # x_val, y_val = shared_dataset((x_val, y_val))
    # note that using shared variables does not work at the moment with L

    # Keep track of which batch we're training with
    batch_idx = 0
    epoch = 0
    while epoch < n_epochs:
        # Extract the training data/label batch and update the parameters with it
        current_x_train = x_train[batch_idx:batch_idx + batch_size]
        current_y_train = y_train[batch_idx:batch_idx + batch_size]
        train(current_x_train, current_y_train)
        batch_idx += batch_size
        # Once we've trained on the entire training set...
        if batch_idx >= x_train.shape[0]:
            # Reset the batch index
            batch_idx = 0
            # Update the number of epochs trained
            epoch += 1
            # Compute the network's output on the validation data
            val_output = net_output_val(x_val)
            train_output = net_output_val(current_x_train)
            # print(val_output[0])
            # print(y_val[0])
            # The accuracy is the average number of correct predictions
            val_error = np.mean(L.objectives.squared_error(val_output, y_val))*100
            train_error = np.mean(L.objectives.squared_error(train_output, current_y_train))*100
            print("Epoch {} validation errors: {} training errors: {}".format(epoch, val_error, train_error))

    # Now we save the model
    np.savez('../data/model.npz', *LL.get_all_param_values(l_output))

    return val_error


def load_my_data(xfile, yfile, n, d, w, valPercent, reduction_size=None):
    def buffered_fetch(fn):
        with open(fn, 'r') as f:
            for line in f:
                yield line

    def create_indices_for_vectors(fn, skip_header=False):
        """
        creates a mapping from the first word on each line to the line number
        useful for retrieving embeddings later for a given word, instead of
        having to store it in memory
        :param fn: fn to create index from
        :param skip_header:
        :param num_examples: the number of words we create indices for
        :return:
        """
        myDict = {}
        word_vectors = []
        count = 0
        for line in buffered_fetch(fn):
            if skip_header:
                skip_header = False
                continue
            splitup = line.rstrip('\n').split(' ')
            token = splitup[0]
            word_vectors.append(np.array(splitup[1:], dtype=np.float32))
            myDict[token] = count
            count += 1
        return myDict, word_vectors

    def get_vector(fn, line_number, offset=0):
        with open(fn, 'r') as f:
            line = list(islice(f, line_number - 1, line_number))[0]
            # islice does not open the entire fn, making it much more
            # memory efficient. the +1 and +2 is because index starts at 0
        v = line.rstrip('\n').split(' ')[1 + offset:]
        # offset needed because there may be spaces or other characters
        # after the first word, but we only want to obtain vectors
        return np.array(list(map(float, v)), dtype=np.float32)

    def load_labels(filename, n_examples, dim):
        data = np.fromfile(filename, dtype=np.float32, count=-1, sep=' ')
        return data.reshape(n_examples, dim)

    def load_vectors(filename, embeddings_file, n_examples, n_words, dim):
        newVec = np.empty([n_examples, 1, n_words, dim], dtype=np.float32)
        word_idx, word_vectors = create_indices_for_vectors(embeddings_file, skip_header=True)
        with open(filename, 'r') as f:
            for n, line in enumerate(f):
                words = line.rstrip('\n').split(' ')
                for m, w in enumerate(words):
                    newVec[n][0][m] = word_vectors[word_idx[w]]
        return newVec

    WORD_EMBEDDINGS = '../data/glove.6B/glove.6B.{0}d.txt'.format(d)
    x_all = load_vectors(xfile, WORD_EMBEDDINGS, n, w, d)
    y_all = load_labels(yfile, n, dim=4096)

    if reduction_size is not None:
        pca_file = '../data/pca_{0}.txt'.format(reduction_size)
        pca_reduction = np.fromfile(pca_file, dtype=np.float32, count=-1, sep=' ')
        pca_reduction = pca_reduction.reshape(4096, reduction_size)
        y_all = normalize(np.dot(y_all, pca_reduction))
    print(y_all.shape)

    np.random.seed(3453)
    randPermute = np.random.permutation(n)
    x_all = np.array([x_all[i] for i in randPermute])
    y_all = np.array([y_all[i] for i in randPermute])
    n_val = int(valPercent * n)

    x_train = x_all[:-n_val]
    x_val = x_all[-n_val:]

    y_train = y_all[:-n_val]
    y_val = y_all[-n_val:]

    return [x_train, y_train, x_val, y_val]


if __name__ == "__main__":

    load_model = False
    try:
        load_option = sys.argv[1]
        if load_option == '-load_model':
            load_model = True
    except IndexError:
        pass

    if load_model:
        print('Loading previously trained model')
    else:
        print('Training fresh model')

    num_examples = 10000
    dim = 200
    num_words = 5
    reduction_size = 200

    training_file = '../data/{0}n_{1}w_training_x.txt'.format(num_examples, num_words)
    truths_file = '../data/{0}n_{1}w_training_gt.txt'.format(num_examples, num_words)

    print "loading data...",
    datasets = load_my_data(training_file, truths_file, n=num_examples, d=dim,
                            w=num_words, valPercent=0.2, reduction_size=reduction_size)
    print "data loaded!"

    results = []
    r = range(0, 10)
    for i in r:
        perf = train_conv_net(datasets,
                              hidden_units=[200, 200, 200],
                              num_filters=[32, 32, 32],
                              filter_hs=[2, 3, 4],
                              n_epochs=100,
                              batch_size=100,
                              dropout_rate=[0.3, 0.5],
                              load_model=load_model)
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)
    print "Final average perf: " + str(np.mean(results))
