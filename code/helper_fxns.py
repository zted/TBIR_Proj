from itertools import islice
import numpy as np


def fetch_most_similar(vect, mat, model, ignore_word=None):
    resultant = vect * np.transpose(mat)
    index = np.argmax(resultant)
    word = model.index2word[index]
    if word == ignore_word:
        resultant[0, index] = -1000
        # ^we do argmax again to get second best word, since first best is itself
        index = np.argmax(resultant)
        word = model.index2word[index]
    return word


def buffered_fetch(fn):
    with open(fn, 'r') as f:
        for line in f:
            yield line


def create_indices_for_vectors(fn, skip_header=False, limit=10000000, return_vectors=False):
    """
    creates a mapping from the first word on each line to the line number
    useful for retrieving embeddings later for a given word, instead of
    having to store it in memory
    :param fn: fn to create index from
    :param skip_header:
    :param limit: the number of words we create indices for
    :return:
    """
    myDict = {}
    word_vectors = []
    count = 0
    for line in buffered_fetch(fn):
        count += 1
        if count > limit:
            break
        if skip_header:
            skip_header = False
            continue
        splitup = line.rstrip('\n').split(' ')
        token = splitup[0]
        myDict[token] = count
        count += 1
        if return_vectors:
            word_vectors.append(np.array(splitup[1:], dtype=np.float32))
    return myDict, word_vectors


def get_vector(fn, line_number, offset=0):
    with open(fn, 'r') as f:
        line = list(islice(f, line_number - 1, line_number))[0]
        # islice does not open the entire fn, making it much more
        # memory efficient. the +1 and +2 is because index starts at 0
    v = line.rstrip('\n').split(' ')[1 + offset:]
    # offset needed because there may be spaces or other characters
    # after the first word, but we only want to obtain vectors
    return np.array(list(map(float, v)))