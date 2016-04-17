from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from itertools import islice
import numpy as np
import time


def bufferdFetch(file):
    with open(file, 'r') as f:
        for line in f:
            yield line


def create_indices_for_vectors(file, skip_header=False, limit=1000000):
    '''
    creates a mapping from the first word on each line to the line number
    useful for retrieving embeddings later for a given word, instead of
    having to store it in memory
    :param file: file to create index from
    :param skip_header:
    :return:
    '''
    myDict = {}
    count = 0
    for line in bufferdFetch(file):
        count += 1
        if count > limit:
            break
        if skip_header:
            skip_header = False
            continue
        token = line.split(' ')[0]
        myDict[token] = count
    return myDict


def unit_vector(v):
    denom = np.linalg.norm(v)
    v = v * 0 if denom == 0 else v / denom
    return v


def get_vector(file, line_number, offset=0):
    with open(file, 'r') as f:
        line = list(islice(f, line_number - 1, line_number))[0]
        # islice does not open the entire file, making it much more
        # memory efficient. the +1 and +2 is because index starts at 0
    v = line.rstrip('\n').split(' ')[1 + offset:]
    # offset needed because there may be spaces or other characters
    # after the first word, but we only want to obtain vectors
    return np.array(list(map(float, v)))


t0 = time.time()
word_dim = 200
num_words = 10
# least number of words needed for each example
number_training_examples = 200

image_embeddings = '../../CLEF/Features/Visual/scaleconcept16_data_visual_vgg16-relu7.dfeat'
word_embeddings = '../data/glove.6B/glove.6B.{0}d.txt'.format(word_dim)
text_training = '../../CLEF/Features/Textual/train_data.txt'

# image_embeddings = '/home/tedz/Desktop/schooldocs/Info Retrieval/' \
#                    'CLEF/Features/Visual/scaleconcept16_data_visual_vgg16-relu7.dfeat'
# word_embeddings = '/home/tedz/Desktop/schooldocs/Info Retrieval/proj/data/glove.6B/glove.6B.200d.txt'
# text_training = '/home/tedz/Desktop/schooldocs/Info Retrieval/CLEF/Features/Textual/train_data.txt'

image_index = create_indices_for_vectors(image_embeddings,
                                         skip_header=True)
word_index = create_indices_for_vectors(word_embeddings,
                                        skip_header=True)

print('Time taken to create word and image indices: ' + str(time.time() - t0))

ignore_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

number_examples_processed = 0
output_x = '../results/{0}n_{1}dim_{2}w_training_x.txt' \
    .format(number_training_examples, word_dim, num_words)
output_y = '../results/{0}n_{1}dim_{2}w_training_gt.txt' \
    .format(number_training_examples, word_dim, num_words)
fx = open(output_x, 'w')
fy = open(output_y, 'w')

with open(text_training, 'r') as f:
    for line in f:
        stemmedWords = set([])
        newArray = np.empty([num_words, word_dim])
        long_string = line.split(' ')
        answer = long_string[0]
        answer_index = image_index[answer]
        answer_vector = get_vector(image_embeddings, answer_index, offset=1)
        total_words = int(len(long_string) / 2)

        if total_words - 1 <= num_words:
            # not enough words, don't bother using this as example
            continue
        count = 0

        for i in range(1, total_words):
            word = long_string[2 * i].split("'")[0]
            # remove apostrophes
            score = long_string[2 * i + 1]

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
                index = word_index[word]
            except KeyError:

                try:
                    index = word_index[lemma]
                except KeyError:

                    try:
                        index = word_index[stem]
                    except KeyError as e:
                        continue

            word_vector = unit_vector(get_vector(word_embeddings, index, 0))
            stemmedWords.add(stem)
            newArray[count] = word_vector.tolist()
            count += 1

            if count >= num_words:
                # we've got enough words, move on to the next example!
                break

        if count >= num_words:
            flattenedArray = newArray.flatten()
            flattenedArray = [str(i) for i in flattenedArray.tolist()]
            answer_vector = [str(i) for i in answer_vector.tolist()]
            fx.write(' '.join(flattenedArray) + '\n')
            fy.write(' '.join(answer_vector) + '\n')
            number_examples_processed += 1
            if number_examples_processed % (number_training_examples / 10) == 0:
                print(str(number_examples_processed) + ' examples processed')
        else:
            continue

        if number_examples_processed >= number_training_examples:
            break

fx.close()
fy.close()
