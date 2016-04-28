import sys

import gensim
import lasagne as L
import lasagne.layers as LL
import numpy as np
import theano
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import helper_fxns as hf


def init_cnn(model_file, hidden_units, num_filters, filter_hs, dropout_rate, n_words, n_dim):
    """
    initializes CNN by loading weights of a previously trained model. note that the model
    trained and this model need to have same parameters. see trainCNN.py for explanation of
    neural network architecture
    :param model_file:
    :param hidden_units:
    :param num_filters:
    :param filter_hs:
    :param dropout_rate:
    :param n_words:
    :param n_dim:
    :return:
    """
    assert len(num_filters) == len(filter_hs)
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((filter_h, n_dim))
        pool_sizes.append((n_words - filter_h + 1, 1))

    l_in = LL.InputLayer(shape=(None, 1, n_words, n_dim))

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

    net_output = theano.function([l_in.input_var], LL.get_output(l_output, deterministic=True))

    with np.load(model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    LL.set_all_param_values(l_output, param_values)

    return net_output


def fetch_top_k(vect, mat, model, k):
    """
    this is different than the one used in retrieve_cnn,
    returns top k doc_IDs that are most similar to the
    input vector
    :param vect:
    :param mat:
    :param model:
    :param k:
    :return:
    """
    resultant = np.dot(mat, vect)
    arglist = np.argsort(resultant)
    arglist = arglist[-1:(-1 - k):-1]
    wordlist = []
    for i in arglist:
        wordlist.append(model.index2word[i])
    return wordlist


def examples_to_vec(test_file, embeddings_file, num_words, word_dim):
    """
    similar function to that used to create training examples. see that
    for full explanation and detailed commented code
    :param test_file:
    :param embeddings_file:
    :param num_words:
    :param word_dim:
    :return: list of input vectors for the CNN, corresponding list of
    the relevant doc_ID
    """
    y = []
    x = []
    ignore_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')
    word_idx, word_vectors = hf.create_indices_for_vectors(embeddings_file, return_vectors=True)
    with open(test_file, 'r') as f:
        for line in f:
            stemmedWords = set([])
            long_string = line.split(' ')
            answer = long_string[0]
            total_words = int(len(long_string) / 2)
            total_example_vec = np.empty([num_words, word_dim], dtype=np.float32)
            if total_words - 1 <= num_words:
                continue
            count = 0

            for i in range(1, total_words):
                word = long_string[2 * i].split("'")[0]

                if (word in ignore_words) or (len(word) <= 3):
                    continue

                if not word.isalpha():
                    continue

                score = long_string[2 * i + 1]

                try:
                    stem = stemmer.stem(word)
                    lemma = lemmatizer.lemmatize(word)
                except UnicodeDecodeError:
                    continue

                if stem in stemmedWords:
                    continue

                try:
                    idx_num = word_idx[word]
                except KeyError:

                    try:
                        idx_num = word_idx[lemma]
                    except KeyError:

                        try:
                            idx_num = word_idx[stem]
                        except KeyError:
                            continue

                word_vec = word_vectors[idx_num]
                total_example_vec[count] = word_vec
                stemmedWords.add(stem)
                count += 1
                if count >= num_words:
                    break

            if count >= num_words:
                y.append(answer)
                x.append(total_example_vec)
    return x, y


def eval_cnn(x, y, num_words, word_dim, cnn_output, gsm_model, topk):
    """
    purpose of this file and function is to evaluate the CNN model against
    a dummy dataset, where we retrieve the top few matches for each example
    and see if we get it right
    :param x: input vectors for the CNN
    :param y: relevant document
    :param num_words: words per example
    :param word_dim:
    :param cnn_output: function that predicts the output using CNN
    :param gsm_model: gensim model
    :param topk: how many results do we want? the more the higher the eval score
    :return:
    """
    bigMatrix = np.array(GENSIM_MODEL.syn0[:], dtype=np.float32)
    n_corr = 0
    n_incorr = 0
    # dividing test data into chunks for evaluation
    batches_x = []
    batches_y = []
    batch_size = 100
    num_batches = len(x) / batch_size
    extra_batch = len(x) % batch_size
    for bn in range(num_batches):
        batches_x.append(x[bn * batch_size:(bn + 1) * batch_size])
        batches_y.append(y[bn * batch_size:(bn + 1) * batch_size])

    if extra_batch != 0:
        batches_x.append(x[num_batches * batch_size:])
        batches_y.append(y[num_batches * batch_size:])
        num_batches += 1

    for i in range(num_batches):
        current_x = batches_x[i]
        current_y = batches_y[i]
        input_vector = np.array(current_x, dtype=np.float32).reshape(len(current_x), 1, num_words, word_dim)
        output_vector = cnn_output(input_vector)
        for n, vec in enumerate(output_vector):
            hypothesis = fetch_top_k(vec, bigMatrix, gsm_model, k=topk)
            if current_y[n] in hypothesis:
                n_corr += 1
            else:
                n_incorr += 1
    print('{} wrong and {} right!'.format(n_incorr, n_corr))
    return


if __name__ == "__main__":

    num_ex_opt = [5, 50, 500, 5000, 10000, 50000]
    topk_opt = [1, 5, 10, 20]

    try:
        num_examples = sys.argv[1]
    except IndexError:
        num_examples = 50

    try:
        CNN_MODEL = sys.argv[2]
    except IndexError:
        CNN_MODEL = '../data/golden_model.npz'

    try:
        TOP_K = sys.argv[3]
    except IndexError:
        TOP_K = 10

    assert (num_examples in num_ex_opt) and (TOP_K in topk_opt)

    WORD_DIM = 200
    NUM_WORDS = 5
    OUTPUT_DIM = 200

    test_y_fn = 'test_y_{}d_{}.txt'.format(OUTPUT_DIM, num_examples)
    TEST_DATA_X = '../data/test_x_{}d_{}.txt'.format(OUTPUT_DIM, num_examples)
    TEST_DATA_Y = '../data/' + test_y_fn
    OUTFILE = '../results/test_results/accuracy_' + test_y_fn
    GENSIM_MODEL = gensim.models.Word2Vec.load_word2vec_format(TEST_DATA_Y, binary=False)
    GENSIM_MODEL.init_sims(replace=True)  # indicates we're finished training to save ram
    WORD_EMBEDDINGS = '../data/glove.6B/glove.6B.{0}d.txt'.format(WORD_DIM)

    VEC_X, VEC_Y = examples_to_vec(TEST_DATA_X, WORD_EMBEDDINGS, NUM_WORDS, WORD_DIM)

    CNN_output = init_cnn(CNN_MODEL,
                          hidden_units=[200, 200, OUTPUT_DIM],
                          num_filters=[32, 32, 32],
                          filter_hs=[2, 3, 4],
                          dropout_rate=[0.3, 0.5],
                          n_words=NUM_WORDS,
                          n_dim=WORD_DIM)

    eval_cnn(VEC_X, VEC_Y, NUM_WORDS, WORD_DIM, CNN_output, GENSIM_MODEL, TOP_K)
