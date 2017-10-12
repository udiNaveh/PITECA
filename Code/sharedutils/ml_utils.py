"""
Utilities used in model learning
"""

import numpy as np
import random


def random_partition(data, k):
    data_copy = ([d for d in data])
    random.shuffle(data_copy)
    return data_copy[:k], data_copy[k:]


class BestWeightsQueue:
    """
    Naive queue implementation to hold the best k sets of weights
    learned in a SGD process. 
    """


    def __init__(self, max_size):
        self.max_size = max_size
        self.dic = dict()
        self.dic[float("infinity")] = None

    def update(self, loss, model_weights):
        if len(self.dic) >= self.max_size:
            if loss < max(self.dic):
                self.dic.pop(max(self.dic))
                self.dic[loss] = model_weights
                return True

            return False
        else:
            self.dic[loss] = model_weights
            return True

    @ property
    def min_key(self):
        return min(self.dic)
    @ property
    def max_key(self):
        return max(self.dic)

    def get_best_weights(self):
        return min(self.dic.items())


class Dataset:

#based on https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data

    def __init__(self,data, labels, additional_data=None):
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self._data = data
        self._labels = labels
        self._num_examples = data.shape[0]
        self._idx = np.arange(0, self._num_examples)
        self._additional_data = additional_data if additional_data is not None else self._idx


    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start == 0:
            np.random.shuffle(self._idx)  # shuffle index
            self._data = self.data[self._idx]  # get list of `num` random samples
            self._labels = self._labels[self._idx]
            self._additional_data = self._additional_data[self._idx]

        # go to the next batch
        elif start + batch_size > self._num_examples:
            self._index_in_epoch=0
            return  self.next_batch(batch_size)

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]
