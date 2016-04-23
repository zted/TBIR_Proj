from itertools import islice
import numpy as np

def buffered_fetch(fn):
    with open(fn, 'r') as f:
        for line in f:
            yield line


def create_indices_for_vectors(fn):
    myDict = {}
    count = 0
    for line in buffered_fetch(fn):
        count += 1
        token = line.split(' ')[0]
        myDict[token] = count
    return myDict


def get_line(fn, line_number):
    with open(fn, 'r') as f:
        line = list(islice(f, line_number - 1, line_number))[0]
    return line

dim = 200
num_examples = 5

image_embeddings = '../data/visfeat_reduced_{}.txt'.format(dim)
image_dict = create_indices_for_vectors(image_embeddings)

indices = np.random.choice(310111, num_examples, False)
training_file = '../data/train_data.txt'
testing_file_x = '../data/test_{}x.txt'.format(num_examples)
testing_file_y = '../data/test_{}y.txt'.format(num_examples)
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
            image_id = line.split(' ')[0]
            idx = image_dict[image_id]
            fo_x.write(line)
            fo_y.write(get_line(image_embeddings, idx))
            count += 1