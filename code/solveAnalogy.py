import linecache as lc
import numpy as np
import scipy.spatial as spat


embeddingsFile = '../data/glove.6B/glove.6B.50d.txt'


def create_indices(file):
    d = {}
    count = 0
    with open(file, mode='r', encoding="ISO-8859-1") as f:
        lines = f.read().splitlines()
        for line in lines:
            count += 1
            word = line.split(' ')[0]
            d[word] = count
    return d


def get_vector(file, line):
    v = lc.getline(file, line)
    v = v.strip('\n').split(' ')[1:]
    return np.array(list(map(float, v)))


def fetch_most_similar(vect, file, ignore_word):
    with open(file, mode='r', encoding="ISO-8859-1") as f:
        best_word = 'yo_mama'
        highest = -1000
        for line in f:
            v = line.strip('\n').split(' ')
            word = v[0]
            if word == ignore_word:
                continue
            v = np.array(list(map(float, v[1:])))
            sim_measure = 1 - spat.distance.cosine(vect, v)
            # cosine similarity
            (best_word, highest) = (word, sim_measure) \
                if sim_measure > highest else (best_word, highest)
    return best_word

wordIndices = create_indices(embeddingsFile)
filename = '../data/questions-words.txt'
count = 0

with open(filename, 'r') as f:
    lines = f.read().splitlines()
    for l in lines:
        words = l.split(' ')
        if words[0] == ':':
            continue
        a = words[0].lower()
        b = words[1].lower()
        c = words[2].lower()
        answer = words[3].lower()
        count += 1
        vecA = get_vector(embeddingsFile, wordIndices[a])
        vecB = get_vector(embeddingsFile, wordIndices[b])
        vecC = get_vector(embeddingsFile, wordIndices[c])
        resultant = vecC + vecB - vecA
        print(fetch_most_similar(resultant, embeddingsFile, c))
        if count > 5:
            break
