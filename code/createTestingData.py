import numpy as np
import helper_fxns as hf

dim = 400
num_examples = 5

image_embeddings = '../data/visfeat_reduced_{}.txt'.format(dim)
image_dict, _ = hf.create_indices_for_vectors(image_embeddings)

indices = np.random.choice(310111, num_examples, False)
# randomly polls from the training data we have to make a test set
# first we create random indices
training_file = '../data/train_data.txt'
testing_file_y = '../data/test_y_{}d_{}.txt'.format(dim, num_examples)
testing_file_x = '../data/test_x_{}.txt'.format(num_examples)
indices.sort()
count = 1
first = True
fo_x = open(testing_file_x, 'w')
fo_y = open(testing_file_y, 'w')
fo_y.write('{} {}\n'.format(num_examples, dim))

with open(training_file, 'r') as f:
    for n, line in enumerate(f):
        if count >= num_examples:
            break
        if first:
            first = False
            continue
        if n-1 == indices[count]:
            # we get the lines at the random indices and get the image ID for that line
            image_id = line.split(' ')[0]
            idx = image_dict[image_id]
            fo_x.write(line)
            # write that line to a file
            fo_y.write(hf.get_line(image_embeddings, idx))
            # write the corresponding line in the image embeddings to a file too
            count += 1