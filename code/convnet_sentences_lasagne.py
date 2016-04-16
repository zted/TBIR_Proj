"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""

import numpy as np
import warnings
from conv_net_classes import *
import lasagne
warnings.filterwarnings("ignore")

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def train_conv_net(datasets,
                   hidden_units,
                   img_w=200,
                   filter_hs=[3,3,3],
                   dropout_rate=0.5,
                   n_epochs=25,
                   batch_size=50):

    x_train, y_train, x_val, y_val, x_test, y_test = datasets

    img_h = len(x_train[0][0])
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((2, 20))
        pool_sizes.append((img_h-filter_h+1, img_w/10))
        #TODO: fix these filter shapes and pool sizes

    input_shape = x_train[0].shape
    l_in = lasagne.layers.InputLayer(
        shape=(None, input_shape[0], input_shape[1], input_shape[2]))


    l_conv1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=filter_shapes[0],
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.HeNormal(gain='relu'))
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2))


    l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, num_filters=32, filter_size=filter_shapes[0],
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.HeNormal(gain='relu'))
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2,2))


    l_conv3 = lasagne.layers.Conv2DLayer(l_pool2, num_filters=32, filter_size=filter_shapes[0],
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.HeNormal(gain='relu'))
    l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, pool_size=(2,2))


    l_hidden1 = lasagne.layers.DenseLayer(l_pool3, num_units=hidden_units[0],
                                          nonlinearity=lasagne.nonlinearities.rectify,
                                          W=lasagne.init.HeNormal(gain='relu'))
    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=dropout_rate)
    l_output = lasagne.layers.DenseLayer(l_hidden1_dropout, num_units=hidden_units[1])
    net_output = lasagne.layers.get_output(l_output)

    true_output = T.matrix('true_output', dtype='float32')

    loss_train = T.mean(lasagne.objectives.squared_error(net_output, true_output))
    all_params = lasagne.layers.get_all_params(l_output)
    # Use ADADELTA for updates
    updates = lasagne.updates.adadelta(loss_train, all_params)
    train = theano.function([l_in.input_var, true_output], loss_train, updates=updates)

    # This is the function we'll use to compute the network's output given an input
    # (e.g., for computing accuracy).  Again, we don't want to apply dropout here
    # so we set the deterministic kwarg to True.
    get_output = theano.function([l_in.input_var],
                             lasagne.layers.get_output(l_output, deterministic=True))

    # Keep track of which batch we're training with
    batch_idx = 0
    # Keep track of which epoch we're onshape
    epoch = 0
    while epoch < n_epochs:
        # Extract the training data/label batch and update the parameters with it
        train(x_train[batch_idx:batch_idx + batch_size],
              y_train[batch_idx:batch_idx + batch_size])
        batch_idx += batch_size
        # Once we've trained on the entire training set...
        if batch_idx >= x_train.shape[0]:
            # Reset the batch index
            batch_idx = 0
            # Update the number of epochs trained
            epoch += 1
            # Compute the network's output on the validation data
            val_output = get_output(x_val)
            # The accuracy is the average number of correct predictions
            error = np.sum(abs(val_output - y_val))/val_output.shape[0]
            print("Epoch {} validation errors: {}".format(epoch, error))

    return error


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
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
    x_val = x_all[n_train:n_train+n_val]
    x_test = x_all[-n_test:]

    y_train = y_all[:n_train]
    y_val = y_all[n_train:n_train+n_val]
    y_test = y_all[-n_test:]

    return [x_train, y_train, x_val, y_val, x_test, y_test]


if __name__=="__main__":

    num_examples = 2000
    dim = 200
    num_words = 20

    training_file = '../data/{0}n_{1}dim_{2}w_training_x.txt'.format(num_examples, dim, num_words)
    truths_file = '../data/{0}n_{1}dim_{2}w_training_gt.txt'.format(num_examples, dim, num_words)

    training_file = '/home/tedz/Desktop/schooldocs/Info Retrieval/' \
                    'proj/data/{0}n_{1}dim_{2}w_training_x.txt'.format(num_examples, dim, num_words)
    truths_file = '/home/tedz/Desktop/schooldocs/Info Retrieval/proj' \
                  '/data/{0}n_{1}dim_{2}w_training_gt.txt'.format(num_examples, dim, num_words)

    print "loading data...",
    datasets = load_my_data(training_file, truths_file, n=num_examples, d=dim,
                            w=num_words, valPercent=0.3, testPercent=0.2)
    print "data loaded!"

    results = []
    r = range(0,10)
    for i in r:
        perf = train_conv_net(datasets,
                              hidden_units=[200,4096],
                              img_w = dim,
                              filter_hs=[3,4,5],
                              n_epochs=10,
                              batch_size=100,
                              dropout_rate=0.5)
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)
    print str(np.mean(results))
