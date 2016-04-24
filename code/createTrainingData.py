from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import sys
import helper_fxns as hf
import numpy as np
import time


def createTrainingExamples(num_training, num_words, output_dim):
    t0 = time.time()
    ignore_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')
    word_dim = 200
    # going to 300 doesn't perform better, 100 performs worse

    IMAGE_EMBEDDINGS = '../data/visfeat_reduced_{}.txt'.format(output_dim)
    image_index, img_vectors = hf.create_indices_for_vectors(IMAGE_EMBEDDINGS, return_vectors=True)

    WORD_EMBEDDINGS = '../data/glove.6B/glove.6B.{0}d.txt'.format(word_dim)
    words_we_have = set([])
    with open(WORD_EMBEDDINGS, 'r') as f:
        for n, line in enumerate(f):
            if n == 0: continue
            token = line.split(' ')[0]
            words_we_have.add(token)

    print('Time taken to load data: ' + str(time.time() - t0))
    t0 = time.time()

    number_examples_processed = 0
    output_x = '../results/{0}n_{1}w_training_x.txt' \
        .format(num_training, num_words)
    output_y = '../results/{0}n_{1}w_training_gt.txt' \
        .format(num_training, num_words)
    fx = open(output_x, 'w')
    fy = open(output_y, 'w')

    TEXT_TRAINING = '../../CLEF/Features/Textual/train_data.txt'
    with open(TEXT_TRAINING, 'r') as f:
        for line in f:
            stemmedWords = set([])
            newArray = np.empty([num_words, word_dim])
            long_string = line.split(' ')
            answer = long_string[0]
            answer_index = image_index[answer]
            answer_vector = img_vectors[answer_index]
            total_words = int(len(long_string) / 2)
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
                    # TODO: see if we should change this to allow alpha numeric
                    continue

                # TODO: remove training examples based on dispersion?

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

                if word in words_we_have:
                    usedWords.append(word)
                elif lemma in words_we_have:
                    usedWords.append(lemma)
                elif stem in words_we_have:
                    usedWords.append(stem)
                else:
                    continue

                stemmedWords.add(stem)
                count += 1

                if count >= num_words:
                    # we've got enough words, move on to the next example!
                    break

            if count >= num_words:
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

    num_train_opt = [2000, 5000, 10000, 50000]
    num_words_opt = [5, 7, 9]
    output_options = [200, 400]

    try:
        num_training = int(sys.argv[1])
    except IndexError:
        num_training = 2000

    try:
        num_words = int(sys.argv[2])
    except IndexError:
        num_words = 5

    try:
        output_dim = int(sys.argv[3])
    except IndexError:
        output_dim = 400

    assert (num_training in num_train_opt and
            num_words in num_words_opt and
            output_dim in output_options)

    createTrainingExamples(num_training, num_words, output_dim)
