import numpy as np
import warnings
from conv_net_classes import *
import lasagne as L
import lasagne.layers as LL

warnings.filterwarnings("ignore")


def train_conv_net(datasets,
                   hidden_units,
                   num_filters=[32, 32, 32],
                   filter_hs=[3, 4, 5],
                   dropout_rate=[0.5],
                   n_epochs=25,
                   batch_size=50):
    x_train, y_train, x_val, y_val, x_test, y_test = datasets

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
                              nonlinearity=L.nonlinearities.rectify,
                              W=L.init.HeNormal(gain='relu'))
    l_hidden1_dropout = LL.DropoutLayer(l_hidden1, p=dropout_rate[0])

    l_output = LL.DenseLayer(l_hidden1_dropout, num_units=hidden_units[1],
                             nonlinearity=L.nonlinearities.linear)
    net_output = LL.get_output(l_output)

    true_output = T.matrix('true_output', dtype='float32')

    loss_train = T.sum(abs(net_output - true_output)) / net_output.shape[0]
    all_params = LL.get_all_params(l_output)
    # Use ADADELTA for updates
    updates = L.updates.adadelta(loss_train, all_params)
    train = theano.function([l_in.input_var, true_output], loss_train, updates=updates)

    # This is the function we'll use to compute the network's output given an input
    # (e.g., for computing accuracy).  Again, we don't want to apply dropout here
    # so we set the deterministic kwarg to True.
    get_output = theano.function([l_in.input_var],
                                 LL.get_output(l_output, deterministic=False))

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
            val_output = get_output(x_val)
            train_output = get_output(current_x_train)
            # print(val_output[0])
            # print(y_val[0])
            # The accuracy is the average number of correct predictions
            val_error = np.sum(abs(val_output - y_val)) / y_val.shape[0]
            train_error = np.sum(abs(train_output - current_y_train)) / train_output.shape[0]
            print("Epoch {} validation errors: {} training errors: {}".format(epoch, val_error, train_error))

    return val_error


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, shared_y


def load_my_data(xfile, yfile, n, d, w, valPercent, testPercent):
    def load_labels(filename, n_examples, dim):
        data = np.fromfile(filename, dtype=np.float32, count=-1, sep=' ')
        return data.reshape(n_examples, dim)

    def load_vectors(filename, n_examples, n_words, dim):
        data = np.fromfile(filename, dtype=np.float32, count=-1, sep=' ')
        return data.reshape(n_examples, 1, n_words, dim)

    x_all = load_vectors(xfile, n, w, d)
    y_all = load_labels(yfile, n, dim=4096)

    np.random.seed(3453)
    randPermute = np.random.permutation(n)
    x_all = np.array([x_all[i] for i in randPermute])
    y_all = np.array([y_all[i] for i in randPermute])
    n_val = int(valPercent * n)
    n_test = int(testPercent * n)
    n_train = n - (n_val + n_test)

    if n_train < 0:
        raise ValueError('Invalid percentages of validation and test data')
    x_train = x_all[:n_train]
    x_val = x_all[n_train:n_train + n_val]
    x_test = x_all[-n_test:]

    y_train = y_all[:n_train]
    y_val = y_all[n_train:n_train + n_val]
    y_test = y_all[-n_test:]

    return [x_train, y_train, x_val, y_val, x_test, y_test]


if __name__ == "__main__":

    num_examples = 2000
    dim = 200
    num_words = 40

    training_file = '../data/{0}n_{1}dim_{2}w_training_x.txt'.format(num_examples, dim, num_words)
    truths_file = '../data/{0}n_training_gt.txt'.format(num_examples)

    print "loading data...",
    datasets = load_my_data(training_file, truths_file, n=num_examples, d=dim,
                            w=num_words, valPercent=0.3, testPercent=0.2)
    print "data loaded!"

    results = []
    r = range(0, 10)
    for i in r:
        perf = train_conv_net(datasets,
                              hidden_units=[1000, 4096],
                              num_filters=[32, 64, 128],
                              filter_hs=[3, 4, 5],
                              n_epochs=20,
                              batch_size=100,
                              dropout_rate=[0.5])
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)
    print str(np.mean(results))
