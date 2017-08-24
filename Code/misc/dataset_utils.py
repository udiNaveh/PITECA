import numpy as np


class Dataset:

#based on https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data

    def __init__(self,data, labels):
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self._data = data
        self._labels = labels
        self._num_examples = data.shape[0]


    @property
    def data(self):
        return self._data

    def next_batch(self,batch_size, use_partial=False):
        start = self._index_in_epoch
        if start == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle index
            self._data = self.data[idx]  # get list of `num` random samples
            self._labels = self._labels[idx]

        # go to the next batch
        elif start + batch_size > self._num_examples:
            self.epochs_completed += 1
            # rest_num_examples = self._num_examples - start
            # data_rest_part = self.data[start:self._num_examples]
            # labels_rest_part = self._labels[start:self._num_examples]
            # idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            # np.random.shuffle(idx0)  # shuffle indexes
            # self._data = self.data[idx0]  # get list of `num` random samples

            self._index_in_epoch=0
            return  self.next_batch(batch_size)
            # self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            # end =  self._index_in_epoch
            # data_new_part =  self._data[start:end]
            # labels_new_part = self._labels[start:end]
            # return np.concatenate((data_rest_part, data_new_part), axis=0), \
            #        np.concatenate((labels_rest_part, labels_new_part), axis=0),


        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]
