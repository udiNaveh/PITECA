import numpy as np


'''
not sure that we will need this file.
'''

def random_partition(array, shape):
    assert type(shape) == list
    if len(array) != sum(shape):
        raise ValueError("shape must sum to length of array")
    inds = []
    for i,part_size in enumerate(shape):
        inds.extend([i] * part_size)
    partition = np.array(inds)
    np.random.shuffle(partition)
    result = []
    for i, part_size in enumerate(shape):
        result.append(array[partition==i])
    return result


