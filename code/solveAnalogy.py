import numpy as np
import time
import gensim
import helper_fxns as hf
from sklearn.preprocessing import normalize
import sys


def cos_sim_addition(a, b, c, matrix, model, ignore_word):
    resultant = c + b - a
    return hf.fetch_most_similar(resultant, matrix, model, ignore_word)


def cos_sim_direction(a, b, c, matrix, model, ignore_word):
    resultant = a - b
    mat = normalize(c - matrix)
    word = hf.fetch_most_similar(resultant, mat, model, ignore_word)
    return word


def cos_sim_multiplication(a, b, c, matrix, model, ignore_word):
    epsi = 0.001
    dc = (np.dot(matrix, c) + 1) / 2
    db = (np.dot(matrix, b) + 1) / 2
    da = (np.dot(matrix, a) + 1) / 2
    resultant = dc * db / (da + epsi)
    index = np.argmax(resultant)
    word = model.index2word[index]
    if word == ignore_word:
        resultant[index] = -1000
        # ^we do argmax again to get second best word, since first best is itself
        index = np.argmax(resultant)
        word = model.index2word[index]
    return word


def analogy_solver(qfile, ofile, gensim_model, lowercase, vocabLimit):
    skipFlag = True
    count = 0
    total_corr = 0
    total_incorr = 0
    n_corr = 0
    n_incorr = 0
    bigMatrix = np.array(gsm_mod.syn0[0:vocabLimit], dtype=np.float32)
    outFile = open(ofile, 'w')
    with open(qfile, 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            words = l.split(' ')
            if lowercase:
                words = [w.lower() for w in words]
            if words[0] == ':':
                if skipFlag == True:
                    # we don't log the first results
                    outFile.write(' '.join(words[1:]) + ':\n')
                    skipFlag = False
                    continue
                recall = float(n_corr * 100) / (n_corr + n_incorr)
                total_recall = float(total_corr * 100) / (total_corr + total_incorr)
                outFile.write('RECALL: {0:.2f} %  ({1:d} / {2:d})\n'
                              .format(recall, n_corr, n_corr + n_incorr))
                outFile.write('Total recall: {0:.2f} %\n'.format(total_recall))
                n_corr = 0
                n_incorr = 0
                outFile.write(' '.join(words[1:]) + ':\n')
                continue
            a = words[0]
            b = words[1]
            c = words[2]
            answer = words[3]
            count += 1
            try:
                vecA = gensim_model[a]
                vecB = gensim_model[b]
                vecC = gensim_model[c]
            except KeyError as e:
                # Couldn't retrieved word in analogy question from our embeddings file
                continue
            hypothesis = a_model(vecA, vecB, vecC, bigMatrix, gensim_model, c)
            # print(hypothesis)
            if hypothesis == answer:
                n_corr += 1
                total_corr += 1
            else:
                n_incorr += 1
                total_incorr += 1
                # if count > 5:
                #     break
        recall = float(n_corr * 100) / (n_corr + n_incorr)
        total_recall = float(total_corr * 100) / (total_corr + total_incorr)
        outFile.write('RECALL: {0:.2f} %  ({1:d} / {2:d})\n'
                      .format(recall, n_corr, n_corr + n_incorr))
        outFile.write('Total recall: {0:.2f} %\n'.format(total_recall))
        t_elapsed = time.time() - t0
        print("Time Taken: ", t_elapsed)
        outFile.close()
    return


if __name__ == "__main__":

    dimension_opt = [50, 100, 200, 300]
    analogy_models = {'addition': cos_sim_addition,
                      'direction': cos_sim_direction,
                      'multiplication': cos_sim_multiplication}
    try:
        word_dim = int(sys.argv[1])
    except IndexError:
        word_dim = 50

    try:
        a_model = sys.argv[2]
    except IndexError:
        a_model = 'addition'

    try:
        vocabLimit = int(sys.argv[3])
    except IndexError:
        vocabLimit = 10000

    assert word_dim in dimension_opt
    a_model = analogy_models[a_model]
    # this is the analogy model that we will be using

    t0 = time.time()
    questionsFile = '../data/questions-words.txt'
    vocabLimit = vocabLimit

    embFileName = 'glove.6B.{0}d.txt'.format(word_dim)
    embeddingsFile = '../data/glove.6B/' + embFileName
    outFile = '../results/accuracy_' + embFileName
    gsm_mod = gensim.models.Word2Vec.load_word2vec_format(embeddingsFile, binary=False)
    gsm_mod.init_sims(replace=True)  # indicates we're finished training to save ram
    makeLowerCase = True

    # embeddingsFile = '../data/GoogleNews-vectors-negative300.bin'
    # outFile = '../results/accuracy_GoogleNews.txt'
    # gsm_mod = gensim.models.Word2Vec.load_word2vec_format(embeddingsFile, binary=True)
    # gsm_mod.init_sims(replace=True)
    # makeLowerCase = False

    analogy_solver(questionsFile, outFile, gsm_mod, makeLowerCase, vocabLimit)
