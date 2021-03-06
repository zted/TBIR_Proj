import sys

import numpy as np

import helper_fxns as hf


def createTestingExamples(num_examples, dim):
    """
    Creates practice examples for us to evaluate how many relevant images
    we can retrieve with our models
    :param num_examples: number of examples to create
    :param dim: visual vector dimensions
    :return:
    """
    image_embeddings = '../data/visfeat_train_reduced_{}.txt'.format(dim)
    image_dict, _ = hf.create_indices_for_vectors(image_embeddings)

    indices = np.random.choice(310111, num_examples, False)
    # randomly polls from the training data we have to make a test set
    # first we create random indices
    training_file = '../data/train_data.txt'
    testing_file_y = '../results/test_y_{}d_{}.txt'.format(dim, num_examples)
    testing_file_x = '../results/test_x_{}d_{}.txt'.format(dim, num_examples)
    indices.sort()
    count = 0
    fo_x = open(testing_file_x, 'w')
    fo_y = open(testing_file_y, 'w')
    fo_y.write('{} {}\n'.format(num_examples, dim))

    with open(training_file, 'r') as f:
        for n, line in enumerate(f):
            if count >= num_examples:
                break
            if n == indices[count]:
                # we get the lines at the random indices and get the image ID for that line
                image_id = line.split(' ')[0]
                idx = image_dict[image_id]
                fo_x.write(line)
                # write that line to a file
                fo_y.write(hf.get_line(image_embeddings, idx))
                # write the corresponding line in the image embeddings to a file too
                count += 1
    return


if __name__ == "__main__":

    num_train_opt = [5, 50, 500, 5000, 10000, 50000]
    output_opt = [200, 400]

    try:
        num_training = int(sys.argv[1])
    except IndexError:
        num_training = 5000

    try:
        output_dim = int(sys.argv[2])
    except IndexError:
        output_dim = 200

    assert (num_training in num_train_opt and
            output_dim in output_opt)
    createTestingExamples(num_training, output_dim)
