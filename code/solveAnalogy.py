import numpy as np
import time
import gensim
import helper_fxns as hf
from sklearn.preprocessing import normalize

t0 = time.time()


def cos_sim_addition(a, b, c, matrix, model, ignore_word):
    resultant = c + b - a
    return hf.fetch_most_similar(resultant, matrix, model, ignore_word)


def cos_sim_direction(a, b, c, matrix, model, ignore_word):
    resultant = a - b
    mat = normalize(c-matrix)
    word = hf.fetch_most_similar(resultant, mat, model, ignore_word)
    return word


def cos_sim_multiplication(a, b, c, matrix, model, ignore_word):
    epsi = 0.001
    dc = (np.dot(matrix, c)+1)/2
    db = (np.dot(matrix, b)+1)/2
    da = (np.dot(matrix, a)+1)/2
    resultant = dc*db/(da + epsi)
    index = np.argmax(resultant)
    word = model.index2word[index]
    if word == ignore_word:
        resultant[index] = -1000
        # ^we do argmax again to get second best word, since first best is itself
        index = np.argmax(resultant)
        word = model.index2word[index]
    return word

questionsFile = '../data/questions-words.txt'
count = 0
total_corr = 0
total_incorr = 0
n_corr = 0
n_incorr = 0
vocabLimit = 10000

embFileName = 'glove.6B.{0}d.txt'.format(50)
embeddingsFile = '../data/glove.6B/' + embFileName
outFile = '../results/accuracy_' + embFileName
model = gensim.models.Word2Vec.load_word2vec_format(embeddingsFile, binary=False)
model.init_sims(replace=True)  # indicates we're finished training to save ram
makeLowerCase = True

# embeddingsFile = '../data/GoogleNews-vectors-negative300.bin'
# outFile = '../results/accuracy_GoogleNews.txt'
# model = gensim.models.Word2Vec.load_word2vec_format(embeddingsFile, binary=True)
# model.init_sims(replace=True)
# makeLowerCase = False

of = open(outFile, 'w')
skipFlag = True
bigMatrix = np.array(model.syn0[0:vocabLimit], dtype=np.float32)

with open(questionsFile, 'r') as f:
    lines = f.read().splitlines()
    for l in lines:
        words = l.split(' ')
        if makeLowerCase:
            words = [w.lower() for w in words]
        if words[0] == ':':
            if skipFlag == True:
                # we don't log the first results
                of.write(' '.join(words[1:]) + ':\n')
                skipFlag = False
                continue
            recall = n_corr * 100 / (n_corr + n_incorr)
            total_recall = total_corr * 100 / (total_corr + total_incorr)
            of.write('RECALL: {0:.2f} %  ({1:d} / {2:d})\n'
                     .format(recall, n_corr, n_corr + n_incorr))
            of.write('Total recall: {0:.2f} %\n'.format(total_recall))
            n_corr = 0
            n_incorr = 0
            of.write(' '.join(words[1:]) + ':\n')
            continue
        a = words[0]
        b = words[1]
        c = words[2]
        answer = words[3]
        count += 1
        try:
            vecA = model[a]
            vecB = model[b]
            vecC = model[c]
        except KeyError as e:
            # Couldn't retrieved word in analogy question from our embeddings file
            continue
        hypothesis = cos_sim_addition(vecA, vecB, vecC, bigMatrix, model, c)
        # print(hypothesis)
        if hypothesis == answer:
            n_corr += 1
            total_corr += 1
        else:
            n_incorr += 1
            total_incorr += 1
            # if count > 5:
            #     break
    recall = n_corr * 100 / (n_corr + n_incorr)
    total_recall = total_corr * 100 / (total_corr + total_incorr)
    of.write('RECALL: {0:.2f} %  ({1:d} / {2:d})\n'
             .format(recall, n_corr, n_corr + n_incorr))
    of.write('Total recall: {0:.2f} %\n'.format(total_recall))
    t_elapsed = time.time() - t0
    print("Time Taken: ", t_elapsed)
    of.close()
