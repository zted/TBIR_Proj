from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import linecache as lc
import numpy as np


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
    myFile = bufferdFetch(file)
    for line in myFile:
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


def get_vector(file, line, first_separator):
    v = lc.getline(file, line)
    v = v.strip('\n')
    v = v.split(first_separator, 1)[-1]
    v = v.split(' ')
    return np.array(list(map(float, v)))


image_embeddings = '/home/tedz/Desktop/schooldocs/Info Retrieval/' \
                   'CLEF/Features/Visual/scaleconcept16_data_visual_vgg16-relu7.dfeat'
word_embeddings = '/home/tedz/Desktop/schooldocs/Info Retrieval/proj/data/glove.6B/glove.6B.200d.txt'
text_training = '/home/tedz/Desktop/schooldocs/Info Retrieval/CLEF/Features/Textual/train_data.txt'

image_embeddings = '../../CLEF/Features/Visual/scaleconcept16_data_visual_vgg16-relu7.dfeat'
word_embeddings = '../data/glove.6B/glove.6B.200d.txt'
text_training = '../../CLEF/Features/Textual/train_data.txt'

image_index = create_indices_for_vectors(image_embeddings,
                                         skip_header=True, limit=2)
word_index = create_indices_for_vectors(word_embeddings,
                                        skip_header=True, limit=30000)
ignore_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
minimum_words = 10
# at least 10 words per example are needed

with open(text_training, 'r') as f:
    stemmedWords = set([])
    addedWords = []
    for line in f:
        long_string = line.split(' ')
        answer = long_string[0]
        answer_index = image_index[answer]
        answer_vector = get_vector(image_embeddings, answer_index, '  ')
        total_words = int(len(long_string)/2)
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
                v = get_vector(word_embeddings, index, ' ')
                stemmedWords.add(stem)
                addedWords.append(word)
                count += 1
            except KeyError as e:
                print("Could not find {0} in dictionary, "
                      "try increasing your vocabulary".format(word))
                continue
            if count >= minimum_words:
                print("we've already got enough words, break!")
                break
            break
        if count >= minimum_words:
            print("we can use this as a training example")
        break
