import numpy as np
import time
import gensim
import helper_fxns as hf
from sklearn.preprocessing import normalize

t0 = time.time()

def cos_sim_addition(a, b, c, matrix, model, ignore_word):
    resultant = c + b - a
    return hf.fetch_most_similar(resultant, matrix, model, ignore_word)


def cos_sim_direction(a, b, c, matrix, file, ignore_word):
    resultant = a - b
    newMatrix = []
    for r in matrix:
        tempArray = c - np.squeeze(np.asarray(r))
        newMatrix.append(normalize(tempArray))
    return hf.fetch_most_similar(resultant, np.asmatrix(newMatrix), file, ignore_word)


def cos_sim_multiplication(a, b, c, matrix, file, ignore_word):
    return


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
bigMatrix = np.asmatrix(model.syn0[0:vocabLimit])
unfound_words = set([])

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
            unfound_words.add(e)
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
    print("Unfound words: ", unfound_words)
    print("Time Taken: ", t_elapsed)
    of.close()
