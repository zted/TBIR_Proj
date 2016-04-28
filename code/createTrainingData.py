import sys
import time

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import helper_fxns as hf


def createTrainingExamples(num_training, num_words, output_dim):
    """
    creates training examples by filtering out stopwords, duplicate words,
    words that are too short, etc. and writes it to a file.
    output file separated into 2, one contains number of words per training example
    other file contains visual feature vectors for the corresponding training example
    the output files are like such
    output_x : word_1, word_2, word_3 ... word_<num_words>
    output_gt: v1_1, v1_2, v1_3, ... v1_<output_dim>
    :param num_training: number of training examples
    :param num_words: number of words to use per example
    :param output_dim: visual feature vector dimensions
    :return:
    """
    t0 = time.time()
    ignore_words = stopwords.words('english')
    # list of stopwords such as 'of, the, a' aka common words that don't carry much meaning
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')
    word_dim = 200

    img_embeddings = '../data/visfeat_train_reduced_{}.txt'.format(output_dim)
    image_index, img_vectors = hf.create_indices_for_vectors(img_embeddings,
                                                             return_vectors=True)

    word_embeddings = '../data/glove.6B/glove.6B.{0}d.txt'.format(word_dim)
    words_we_have = set([])
    with open(word_embeddings, 'r') as f:
        for n, line in enumerate(f):
            if n == 0:
                continue
            token = line.split(' ')[0]
            words_we_have.add(token)
            # creates a set containing all the words we have embeddings for

    print('Time taken to load data: ' + str(time.time() - t0))
    t0 = time.time()

    number_examples_processed = 0
    output_x = '../results/{0}n_{1}w_training_x.txt' \
        .format(num_training, num_words)
    output_y = '../results/{0}n_{1}w_{2}d_training_gt.txt' \
        .format(num_training, num_words, output_dim)
    fx = open(output_x, 'w')
    fy = open(output_y, 'w')

    TEXT_TRAINING = '../data/train_data.txt'
    with open(TEXT_TRAINING, 'r') as f:
        for line in f:
            # for each line we keep track of the stems of each word we use
            stemmedWords = set([])
            long_string = line.split(' ')
            answer = long_string[0]
            answer_index = image_index[answer]
            answer_vector = img_vectors[answer_index]
            # the visual feature vector corresponding to this example's imageID
            total_words = int(len(long_string) / 2)
            # training data tells us how many words this example has
            usedWords = []
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
                    # use stemmer and lemmatizer to raise chances of finding the
                    # word's embedding, e.g. hippopotamuses may not be in the
                    # word embeddings file but hippopotamus may be
                except UnicodeDecodeError:
                    # sometimes the word has non-english letters, such as the 2
                    # dots over the O or a tilda above it
                    continue

                if stem in stemmedWords:
                    # we've already used a variant of the word in this example
                    # i.e. we do not want to use both words run and running
                    continue

                if word in words_we_have:
                    usedWords.append(word)
                elif lemma in words_we_have:
                    usedWords.append(lemma)
                elif stem in words_we_have:
                    usedWords.append(stem)
                else:
                    # we do not have embeddings for the word
                    continue

                # word passed all the filters
                stemmedWords.add(stem)
                count += 1

                if count >= num_words:
                    # we've got enough words, move on to the next example!
                    break

            if count >= num_words:
                # if we have enough words for this example, write the words and
                # ground truth to their respective files. if not, do nothing
                answer_vector = [str(i) for i in answer_vector.tolist()]
                fx.write(' '.join(usedWords) + '\n')
                fy.write(' '.join(answer_vector) + '\n')
                number_examples_processed += 1
                if number_examples_processed % (num_training / 10) == 0:
                    print("{} seconds taken to create {} training examples."
                          .format(time.time() - t0, number_examples_processed))
            else:
                continue

            if number_examples_processed >= num_training:
                break
    fx.close()
    fy.close()
    return


if __name__ == "__main__":

    num_train_opt = [2000, 10000, 50000, 200000, 300000]
    num_words_opt = [5, 7, 9]
    # can use 5, 7, or 9 words per example
    output_options = [200, 400]
    # can output feature vectors in 200dim or 400dim depending on
    # past performance test results
    try:
        NUM_TRAINING = int(sys.argv[1])
    except IndexError:
        NUM_TRAINING = 2000

    try:
        NUM_WORDS = int(sys.argv[2])
    except IndexError:
        NUM_WORDS = 5

    try:
        OUTPUT_DIM = int(sys.argv[3])
    except IndexError:
        OUTPUT_DIM = 200

    assert (NUM_TRAINING in num_train_opt and
            NUM_WORDS in num_words_opt and
            OUTPUT_DIM in output_options)

    createTrainingExamples(NUM_TRAINING, NUM_WORDS, OUTPUT_DIM)
