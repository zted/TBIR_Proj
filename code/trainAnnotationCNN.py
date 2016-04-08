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
    v = v*0 if denom == 0 else v/denom
    return v


def get_vector(file, line_number, offset=0):
    with open(file, 'r') as f:
        line = list(islice(f,line_number-1,line_number))[0]
        # islice does not open the entire file, making it much more
        # memory efficient. the +1 and +2 is because index starts at 0
    v = line.rstrip('\n').split(' ')[1+offset:]
    # offset needed because there may be spaces or other characters
    # after the first word, but we only want to obtain vectors
    return np.array(list(map(float, v)))

t0 = time.time()
image_embeddings = '../../CLEF/Features/Visual/scaleconcept16_data_visual_vgg16-relu7.dfeat'
word_embeddings = '../data/glove.6B/glove.6B.200d.txt'
text_training = '../../CLEF/Features/Textual/train_data.txt'

# image_embeddings = '/home/tedz/Desktop/schooldocs/Info Retrieval/' \
#                    'CLEF/Features/Visual/scaleconcept16_data_visual_vgg16-relu7.dfeat'
# word_embeddings = '/home/tedz/Desktop/schooldocs/Info Retrieval/proj/data/glove.6B/glove.6B.200d.txt'
# text_training = '/home/tedz/Desktop/schooldocs/Info Retrieval/CLEF/Features/Textual/train_data.txt'

image_index = create_indices_for_vectors(image_embeddings,
                                         skip_header=True, limit=3)
word_index = create_indices_for_vectors(word_embeddings,
                                        skip_header=True, limit=30000)
ignore_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
minimum_words = 10
# at least 10 words per example are needed

t = time.time()
print("T1: ", t-t0)

with open(text_training, 'r') as f:
    print("Time to open training data: ", time.time() - t)
    t = time.time()
    stemmedWords = set([])
    addedWords = []
    for line in f:
        long_string = line.split(' ')
        answer = long_string[0]
        answer_index = image_index[answer]
        answer_vector = get_vector(image_embeddings, answer_index, offset=1)
        total_words = int(len(long_string)/2)
        print("T2: ", time.time() - t)
        t = time.time()
        if total_words-1 <= minimum_words:
            # not enough words, don't bother using this as example
            continue
        count = 0
        for i in range(1, total_words):
            word = long_string[2*i].split("'")[0]
            # remove apostrophes
            score = long_string[2*i+1]
            stem = stemmer.stem(word)
            if stem in stemmedWords:
                # we've used a variant of this word already
                continue
            lemma = lemmatizer.lemmatize(word)
            # use lemma to find word easily
            try:
                index = word_index[lemma]
                word_vector = get_vector(word_embeddings, index, 0)
                stemmedWords.add(stem)
                addedWords.append(word)
                count += 1
            except KeyError as e:
                print("Could not find {0} in dictionary, "
                      "try increasing your vocabulary".format(word))
                continue
            print("T3: ", time.time() - t)
            t = time.time()
            if count >= minimum_words:
                print("we've already got enough words, break!")
                break
            break
        if count >= minimum_words:
            print("we can use this as a training example")
        break
