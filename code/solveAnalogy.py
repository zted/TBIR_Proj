import linecache as lc
import numpy as np
import time
import word2vec as w2v
t0 = time.time()


def cosine_similarity(u,v):
    numerator = np.dot(u,v)
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    denominator = denominator if denominator != 0 else 0
    # ^do not divide by 0!
    return numerator/denominator


def construct_matrix(model, limit):
    l = []
    for n, v in enumerate(model.vectors):
        l.append(v)
        if n >= limit-1:
            return np.asmatrix(l, float)
            # limit number of vocabulary we loop over to save time
    return np.asmatrix(l, float)


def fetch_most_similar(vect, mat, file, ignore_word):
    resultant = vect * np.transpose(mat)
    index = np.argmax(resultant)
    v = lc.getline(file, index+2)
    word = v.split(' ')[0]
    if word == ignore_word:
        resultant[0,index] = -100
        # ^we do argmax again to get second best word, since first best is itself
        index = np.argmax(resultant)
        v = lc.getline(file, index+2)
        word = v.split(' ')[0]
    return word


def cos_sim_addition(a, b, c, matrix, file, ignore_word):
    resultant = c+b-a
    return fetch_most_similar(resultant, matrix, file, ignore_word)


# def cos_sim_direction(a, b, c, matrix, file, ignore_word):
#     resultant = a-b
#     newMatrix = []
#     for r in matrix:
#         tempArray = c-np.squeeze(np.asarray(r))
#         mag = np.linalg.norm(tempArray)
#         tempArray = tempArray * 0 if mag == 0 else tempArray/mag
#         newMatrix.append((tempArray)/np.linalg.norm(tempArray))
#     print(newMatrix)
#     return fetch_most_similar(resultant, np.asmatrix(newMatrix), file, ignore_word)


def cos_sim_multiplication(a, b, c, matrix, file, ignore_word):
    return

embFileName = 'glove.6B.{0}d.txt'.format(300)
embeddingsFile = '../data/glove.6B/' + embFileName
filename = '../data/questions-words.txt'
count = 0
total_corr = 0
total_incorr = 0
n_corr = 0
n_incorr = 0
vocabLimit = 10000
outFile = '../results/accuracy_' + embFileName
embeddingsFile = '../data/GoogleNews-vectors-negative300.bin'
outFile = '../results/accuracy_GoogleNews.txt'
model = w2v.load(embeddingsFile, 'bin')
of = open(outFile, 'w')
skipFlag = True
bigMatrix = construct_matrix(model, vocabLimit)

with open(filename, 'r') as f:
    lines = f.read().splitlines()
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
        try:
            vecA = model[a]
            vecB = model[b]
            vecC = model[c]
        except KeyError as e:
            # Couldn't retrieved word in analogy question from our embeddings file
            print(e)
            continue
        hypothesis = cos_sim_addition(vecA, vecB, vecC, bigMatrix, embeddingsFile, c)
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
