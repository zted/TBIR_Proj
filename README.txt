The NN architecture was inspired by and adapted from Yoon Kim's use of CNN
for sentence classification.
  Code https://github.com/yoonkim/CNN_sentence
  and paper http://arxiv.org/abs/1408.5882

----------------------------------------------------------------------------------------
To run document retrieval on a file containing queries, cd into code and type
       python retrieve_cnn.py path/to/file path/to/cnn_model
The first argument must be provided. It is a file containing all the queries
in the format of the CLEF 2016 teaser1 test set. If the second model is not
provided, the code will automatically look for a pretrained model in data/
named golden_model.npz. The output file is results/rank_predictions.txt.

golden_model.npz is trained on a dataset of roughly 300,000 examples. Each
example contains 5 words, and its corresponding 200 dimension vector. Refer
to trainCNN.py to see what settings the network was trained on.

----------------------------------------------------------------------------------------
The files that must be present for the code to run are data/visfeat_test_reduced_{x}.txt,
and data/glove.6B/glove.6B.{y}d.txt, where x and y are dimension numbers.

The first file contains the list of documents we are choosing from, with each document
being represented with a vector of x dimension. This file was obtained by running PCA
on a sample of 100000 other feature representation vectors of 4096 dimensions, then
applying the resulting PCA to our dataset. Furthermore, the file contains only document
IDs for which we don't know what their corresponding words are, as this is the test set.
For evaluation I am using 200 dimension vectors, this makes retrieval top 100 documents
fairly quick.

The second file contains word embeddings. For evaluation I use 200 dimension word vectors,
as the results are equally bad for other ones. To use different dimension word vectors,
one would need to train the cnn on 200 dimensions.

----------------------------------------------------------------------------------------
Summary of what each file does:
createTestingData: creates dummy testing data to evaluate our CNN ranking predictions
against training examples
createTrainingData: creates training data to train our CNN
evaluate_cnn: takes dummy testing data and outputs top_k predictions
helper_fxns: contains commonly used functions
retrieve_cnn: outputs rankings based on queries
solveAnalogy: solves word analogies using word embedding based on different models
trainCNN: trains the convolution neural network
PCA.m: finds k principal components from the original 4096 dimensioned visual feature vectors
