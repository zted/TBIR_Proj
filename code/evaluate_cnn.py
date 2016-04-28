import gensim
import lasagne as L
import lasagne.layers as LL
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import helper_fxns as hf
import numpy as np
import theano


def init_cnn(model_file, hidden_units, num_filters, filter_hs, dropout_rate, n_words, n_dim):
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
    resultant = np.dot(mat, vect)
    arglist = np.argsort(resultant)
    arglist = arglist[-1:(-1-k):-1]
    wordlist = []
    for i in arglist:
        wordlist.append(model.index2word[i])
    return wordlist


num_examples = 10000
word_dim = 200
num_words = 5
output_dim = 200

test_y_fn = 'test_y_{}d_{}.txt'.format(output_dim, num_examples)
TEST_DATA_X = '../data/test_x_{}.txt'.format(num_examples)

TEST_DATA_Y = '../data/' + test_y_fn
outFile = '../results/test_results/accuracy_' + test_y_fn
gensim_model = gensim.models.Word2Vec.load_word2vec_format(TEST_DATA_Y, binary=False)
gensim_model.init_sims(replace=True)  # indicates we're finished training to save ram
makeLowerCase = True

WORD_EMBEDDINGS = '../data/glove.6B/glove.6B.{0}d.txt'.format(word_dim)
word_idx, word_vectors = hf.create_indices_for_vectors(WORD_EMBEDDINGS, return_vectors=True)
count = 0
n_corr = 0
n_incorr = 0

ignore_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

# of = open(outFile, 'w')
bigMatrix = np.array(gensim_model.syn0[:], dtype=np.float32)

CNN_MODEL = '../data/golden_model.npz'
CNN_predict = init_cnn(CNN_MODEL,
                       hidden_units=[300, 300, 400],
                       num_filters=[32, 32, 32],
                       filter_hs=[2, 3, 4],
                       dropout_rate=[0.3, 0.5],
                       n_words=num_words,
                       n_dim=word_dim)

y = []
x = []
with open(TEST_DATA_X, 'r') as f:
    for line in f:
        stemmedWords = set([])
        newArray = np.empty([num_words, word_dim])
        long_string = line.split(' ')
        answer = long_string[0]
        total_words = int(len(long_string) / 2)
        total_example_vec = np.empty([num_words, word_dim], dtype=np.float32)
        if total_words - 1 <= num_words:
            # not enough words, don't bother using this as example
            continue
        count = 0

        for i in range(1, total_words):
            word = long_string[2 * i].split("'")[0]
            # take out the apostrophe from words first

            if (word in ignore_words) or (len(word) <= 3):
                # ignore the stopwords
                continue

            if not word.isalpha():
                # ignore words that are not purely alphabets, so 76ers, etc
                continue

            score = long_string[2 * i + 1]
            # we may add this information later into the word vector

            try:
                stem = stemmer.stem(word)
                lemma = lemmatizer.lemmatize(word)
                # use lemma to find word easily
            except UnicodeDecodeError as e:
                # print('Could not stem or lemmatize ' + word)
                continue

            if stem in stemmedWords:
                # we've already used a variant of the word in this example
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
                        # word simply cannot be found in our embeddings file
                        continue

            word_vec = word_vectors[idx_num]
            total_example_vec[count] = word_vec
            stemmedWords.add(stem)
            count += 1
            if count >= num_words:
                # we've got enough words, move on to the next example!
                break

        if count >= num_words:
            y.append(answer)
            x.append(total_example_vec)

    # dividing test data into chunks for evaluation
    batches_x = []
    batches_y = []
    batch_num = 0
    batch_size = 10
    num_batches = len(x) / 10
    extra_batch = len(x) % 10
    for bn in range(num_batches):
        batches_x.append(x[bn * batch_size:(bn + 1) * batch_size])
        batches_y.append(y[bn * batch_size:(bn + 1) * batch_size])

    if extra_batch != 0:
        batches_x.append(x[num_batches*batch_size:])
        batches_y.append(y[num_batches*batch_size:])
        num_batches += 1

    for i in range(num_batches):
        current_x = batches_x[i]
        current_y = batches_y[i]
        input_vector = np.array(current_x, dtype=np.float32).reshape(len(current_x), 1, num_words, word_dim)
        output_vector = CNN_predict(input_vector)
        for n, vec in enumerate(output_vector):
            hypothesis = fetch_top_k(vec, bigMatrix, gensim_model, k=100)
            if current_y[n] in hypothesis:
                print('Correct answer!')
                n_corr += 1
            else:
                n_incorr += 1
        print('Next batch')
    print('{} wrong and {} right!'.format(n_incorr, n_corr))
