"""
The architecture was inspired by and adapted from Yoon
Kim's use of CNN for sentence classification.
  Code https://github.com/yoonkim/CNN_sentence
  and paper http://arxiv.org/abs/1408.5882
"""
import sys
import warnings

import lasagne as L
import lasagne.layers as LL
import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import normalize

import helper_fxns as hf

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

    # creating filter and pool sizes
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

    # making convolution and pooling layers
    layer_list = []
    for i in range(len(filter_hs)):
        # by using a for loop it will have as many convolutions as there are filters
        l_conv = LL.Conv2DLayer(l_in, num_filters=num_filters[i], filter_size=filter_shapes[i],
                                nonlinearity=L.nonlinearities.rectify,
                                W=L.init.HeNormal(gain='relu'))
        l_pool = LL.MaxPool2DLayer(l_conv, pool_size=pool_sizes[i])
        layer_list.append(l_pool)

    mergedLayer = LL.ConcatLayer(layer_list)

    # hidden and dropout layers
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

    if load_model:
        # if option is selected, load previously trained model
        with np.load('../data/model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        LL.set_all_param_values(l_output, param_values)

    # output layers, for training we use stochastic determination of dropout
    net_output_train = LL.get_output(l_output, deterministic=False)

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
            val_error = np.mean(L.objectives.squared_error(val_output, y_val)) * len(y_val[0])
            train_error = np.mean(L.objectives.squared_error(train_output, current_y_train)) * len(y_val[0])
            print("Epoch {} validation errors: {} training errors: {}".format(epoch, val_error, train_error))

    # Now we save the model
    np.savez('../data/model.npz', *LL.get_all_param_values(l_output))

    return val_error


def load_my_data(xfile, yfile, n, d, w, output_d, valPercent):
    """
    loads data, converts the words in xfile to vectors by looking up
    its corresponding embeddings file
    :param xfile: file containing words and word embeddings
    :param yfile: file containing visual feature representations
    :param n: number of training examples
    :param d: dimension of each word embedding
    :param w: number of words per example
    :param output_d: dimension of output, or visual feature embedding dimension
    :param valPercent: percentage of training set used for validation
    :return: training and validation data shuffled
    """

    def load_labels(filename, n_examples, dim):
        data = np.fromfile(filename, dtype=np.float32, count=-1, sep=' ')
        return data.reshape(n_examples, dim)

    def load_vectors(filename, embeddings_file, n_examples, n_words, dim):
        newVec = np.empty([n_examples, 1, n_words, dim], dtype=np.float32)
        word_idx, word_vectors = hf.create_indices_for_vectors(embeddings_file, return_vectors=True)
        with open(filename, 'r') as f:
            for n, line in enumerate(f):
                words = line.rstrip('\n').split(' ')
                for m, w in enumerate(words):
                    newVec[n][0][m] = word_vectors[word_idx[w]]
        return newVec

    word_embeddings = '../data/glove.6B/glove.6B.{0}d.txt'.format(d)
    x_all = load_vectors(xfile, word_embeddings, n, w, d)
    y_all = normalize(load_labels(yfile, n, output_d))

    np.random.seed(3453)
    randPermute = np.random.permutation(n)
    # gives us a random permutation of the data so that we get different
    # validation and training data in different batches each time
    x_all = np.array([x_all[i] for i in randPermute])
    y_all = np.array([y_all[i] for i in randPermute])
    n_val = int(valPercent * n)

    x_train = x_all[:-n_val]
    x_val = x_all[-n_val:]

    y_train = y_all[:-n_val]
    y_val = y_all[-n_val:]

    return [x_train, y_train, x_val, y_val]


if __name__ == "__main__":
    # select number of examples to train on
    examples_options = [2000, 10000, 50000, 200000, 300000]
    load_model = False

    # load previously trained model or train new one
    try:
        load_option = sys.argv[1]
        if load_option == '-load_model':
            load_model = True
    except IndexError:
        pass

    try:
        num_examples = int(sys.argv[2])
    except IndexError:
        num_examples = 2000

    assert num_examples in examples_options

    if load_model:
        print('Loading previously trained model')
    else:
        print('Training fresh model')

    WORD_DIM = 200
    # dimension of each word embedding
    NUM_WORDS = 5
    # number of words per training example
    OUTPUT_DIM = 200

    TRAINING_FILE = '../data/{0}n_{1}w_training_x.txt'.format(num_examples, NUM_WORDS)
    TRUTHS_FILE = '../data/{0}n_{1}w_{2}d_training_gt.txt'.format(num_examples, NUM_WORDS, OUTPUT_DIM)

    print "loading data...",
    datasets = load_my_data(TRAINING_FILE, TRUTHS_FILE, n=num_examples, d=WORD_DIM,
                            w=NUM_WORDS, output_d=OUTPUT_DIM, valPercent=0.2)
    print "data loaded!"

    results = []
    r = range(0, 1)
    for i in r:
        # note that the last hidden layer must have same dimension as the output
        perf = train_conv_net(datasets,
                              hidden_units=[200, 200, OUTPUT_DIM],
                              num_filters=[32, 32, 32],
                              filter_hs=[2, 3, 4],
                              n_epochs=50,
                              batch_size=100,
                              dropout_rate=[0.3, 0.5],
                              load_model=load_model)
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)
    print "Final average perf: " + str(np.mean(results))
