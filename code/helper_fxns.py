"""
This file contains a few commonly used functions
"""
from itertools import islice

import numpy as np


def fetch_most_similar(vect, mat, model, ignore_word=None):
    """
    uses cosine similarity to find the most similar word
    :param vect:
    :param mat:
    :param model:
    :param ignore_word: a-b is to c-d, if we want to find d,
    we should ignore the answer if it is c
    :return:
    """
    resultant = np.dot(mat, vect)
    index = np.argmax(resultant)
    word = model.index2word[index]
    if word == ignore_word:
        resultant[index] = -1000
        # ^we do argmax again to get second best word, since first best is itself
        index = np.argmax(resultant)
        word = model.index2word[index]
    return word


def buffered_fetch(fn):
    with open(fn, 'r') as f:
        for line in f:
            yield line


def create_indices_for_vectors(fn, limit=10000000, return_vectors=False):
    """
    creates a mapping from the first word on each line to the line number
    in a file, or if we want list of vectors to be returned, the mapped value
    corresponds to the nth vector
    :param return_vectors: whether or not we want vectors returned
    :param fn: fn to create index from
    :param limit: the number of words we create indices for
    :return:
    """
    myDict = {}
    count = 0
    first = True
    count_offset = 2 if return_vectors else 0
    for line in buffered_fetch(fn):
        count += 1
        if count > limit:
            break
        if first:
            vocabsize = int(line.split(' ')[0])
            dim = int(line.rstrip('\n').split(' ')[1])
            word_vectors = np.empty([vocabsize, dim])
            first = False
            continue
        splitup = line.rstrip('\n').split(' ')
        token = splitup[0]
        myDict[token] = count - count_offset
        if return_vectors:
            word_vectors[count - count_offset] = np.array(splitup[1:], dtype=np.float32)
    return myDict, word_vectors


def get_line(fn, line_number):
    """
    gets the line of a file
    :param fn:
    :param line_number:
    :return:
    """
    with open(fn, 'r') as f:
        line = list(islice(f, line_number - 1, line_number))[0]
    return line
