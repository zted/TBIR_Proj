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
from collections import OrderedDict
import theano.tensor as T
import warnings
import time
from conv_net_classes import *
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
                   filter_hs=[2,3,4,5],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    x_train, y_train, x_val, y_val, x_test, y_test = datasets

    rng = np.random.RandomState(3435)
    img_h = len(x_train[0])
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))

    #define model architecture
    index = T.lscalar()
    x = T.tensor3('x', dtype='float32')
    y = T.matrix('y', dtype='float32')
    layer0_input = x.reshape((x.shape[0], 1, x.shape[1], img_w))

    conv_layers = []
    layer1_inputs = []
    for i in range(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs, axis=1)
    hidden_units[0] = feature_maps*len(filter_hs)
    classifier = Dropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)

    #define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    cost = classifier.errors(y)
    dropout_cost = classifier.dropout_errors(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    n_batches = x_train.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    n_val_batches = n_batches - n_train_batches
    #divide train set into train/val sets

    x_train, y_train = shared_dataset((x_train,y_train))
    x_val, y_val = shared_dataset((x_val,y_val))
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: x_val[index * batch_size: (index + 1) * batch_size],
             y: y_val[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
    # compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: x_train[index * batch_size: (index + 1) * batch_size],
                 y: y_train[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: x_train[index*batch_size:(index+1)*batch_size],
              y: y_train[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)

    test_pred_layers = []
    test_size = x_test.shape[0]
    test_layer0_input = x.reshape((test_size,1,img_h,img_w))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.sum(abs(test_y_pred - y))
    test_model_all = theano.function([x,y], test_error, allow_input_downcast = True)
    show_prediction = theano.function([x], test_y_pred, allow_input_downcast = True)

    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0
    cost_epoch = 0
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
        else:
            for minibatch_index in range(n_train_batches):
                cost_epoch = train_model(minibatch_index)
        train_losses = [test_model(i) for i in range(n_train_batches)]
        train_perf = np.mean(train_losses)
        val_losses = [val_model(i) for i in range(n_val_batches)]
        val_perf = np.mean(val_losses)
        # print(show_prediction(x_test))
        # print(np.asarray(show_prediction).shape)
        print('epoch: %i, training time: %.2f secs, train errors: %.2f, val errors: %.2f' % (epoch, time.time()-start_time, train_perf, val_perf))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = test_model_all(x_test,y_test)
            test_perf = 1- test_loss
    return test_perf


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


def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


def load_my_data(xfile, yfile, n, d, w, valPercent, testPercent):

    def load_labels(filename, n_examples, dim):
        data = np.fromfile(filename, dtype=np.float32, count=-1, sep=' ')
        return data.reshape(n_examples, dim)

    def load_vectors(filename, n_examples, n_words, dim):
        data = np.fromfile(filename, dtype=np.float32, count=-1, sep=' ')
        return data.reshape(1, n_examples, n_words, dim)

    x_all = load_vectors(xfile, n, w, d)[0]
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

    # training_file = '/home/tedz/Desktop/schooldocs/Info Retrieval/' \
    #                 'proj/data/{0}n_{1}dim_{2}w_training_x.txt'.format(num_examples, dim, num_words)
    # truths_file = '/home/tedz/Desktop/schooldocs/Info Retrieval/proj' \
    #               '/data/{0}n_{1}dim_{2}w_training_gt.txt'.format(num_examples, dim, num_words)

    print "loading data...",
    datasets = load_my_data(training_file, truths_file, n=num_examples, d=dim,
                            w=num_words, valPercent=0.3, testPercent=0.2)
    print "data loaded!"

    results = []
    r = range(0,10)
    for i in r:
        perf = train_conv_net(datasets,
                              hidden_units=[200,4096],
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              shuffle_batch=True,
                              n_epochs=5,
                              sqr_norm_lim=9,
                              batch_size=100,
                              dropout_rate=[0.5])
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)  
    print str(np.mean(results))
