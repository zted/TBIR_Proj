import linecache as lc
import numpy as np
import time


def cosine_similarity(u,v):
    numerator = np.dot(u,v)
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    denominator = denominator if denominator != 0 else 0
    return numerator/denominator


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


def construct_matrix(file, limit):
    with open(file, mode='r', encoding="ISO-8859-1") as f:
        l = []
        for n, line in enumerate(f):
            v = line.strip('\n').split(' ')
            v = np.array(list(map(float, v[1:])))
            l.append(v/np.linalg.norm(v))
            if n >= limit-1:
                return np.asmatrix(l, float)
                # limit number of vocabulary we loop over to save time
    return l


def fetch_most_similar(vect, mat, file, ignore_word):
    resultant = vect * np.transpose(mat)
    index = np.argmax(resultant)
    v = lc.getline(file, index+1)
    word = v.split(' ')[0]
    if word == ignore_word:
        resultant[0,index] = -100
        index = np.argmax(resultant)
        v = lc.getline(file, index+1)
        word = v.split(' ')[0]
    return word


embFileName = 'glove.6B.{0}d.txt'.format(50)
embeddingsFile = '../data/glove.6B/' + embFileName
wordIndices = create_indices(embeddingsFile)
filename = '../data/questions-words.txt'
count = 0
total_corr = 0
total_incorr = 0
n_corr = 0
n_incorr = 0
vocabLimit = 10000
outFile = '../results/accuracy_' + embFileName
of = open(outFile, 'w')
skipFlag = True
bigMatrix = construct_matrix(embeddingsFile, vocabLimit)

with open(filename, 'r') as f:
    lines = f.read().splitlines()
    t0 = time.time()
    for l in lines:
        words = l.split(' ')
        if words[0] == ':':
            if skipFlag == True:
                # we don't log the first results
                of.write(' '.join(words[1:]) + ':\n')
                skipFlag = False
                continue
            recall = n_corr*100/(n_corr+n_incorr)
            total_recall = total_corr*100/(total_corr+total_incorr)
            of.write('RECALL: {0:.2f} %  ({1:d} / {2:d})\n'
                     .format(recall, n_corr, n_corr+n_incorr))
            of.write('Total recall: {0:.2f} %\n'.format(total_recall))
            n_corr = 0
            n_incorr = 0
            of.write(' '.join(words[1:]) + ':\n')
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
        hypothesis = fetch_most_similar(resultant, bigMatrix, embeddingsFile, c)


        # print(hypothesis)
        if hypothesis == answer:
            n_corr += 1
            total_corr += 1
        else:
            n_incorr += 1
            total_incorr += 1
        # if count > 5:
        #     break
    recall = n_corr*100/(n_corr+n_incorr)
    total_recall = total_corr*100/(total_corr+total_incorr)
    of.write('RECALL: {0:.2f} %  ({1:d} / {2:d})\n'
                .format(recall, n_corr, n_corr+n_incorr))
    of.write('Total recall: {0:.2f} %\n'.format(total_recall))
    t_elapsed = time.time() - t0
    print("Time Taken: ", t_elapsed)
    of.close()
