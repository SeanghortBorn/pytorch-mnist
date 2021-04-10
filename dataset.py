from random import shuffle
import cv2
from os import listdir
import torch

class DataSet(object):
    def __init__(self, path, num_class):
        x = list()
        y = list()
        for c in range(num_class):
            tmp = listdir(path + str(c) + '/')
            for file in tmp:
                x.append(cv2.imread(path + str(c) + '/' + file, cv2.IMREAD_GRAYSCALE))  # np (H, W)
                y.append(c)

        self._data = x
        self._labels = y
        self._num_examples = len(self._data)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data # list of np (H, W)

    @property
    def labels(self):
        return self._labels #list of int

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def reset(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def randomize(self):
        tmp_shuf = list(zip(self._data, self._labels))
        shuffle(tmp_shuf)
        self._data, self._labels = zip(*tmp_shuf)
        self._data = list(self._data)
        self._labels = list(self._labels)

    def next_batch(self, batch_size, shuffle=True, equal_last_batch=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            self.randomize()
        # Go to the next epoch
        if start + batch_size >= self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                self.randomize()
            # Clip last batch
            if equal_last_batch:
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch
                data_new_part = self._data[start:end]
                labels_new_part = self._labels[start:end]
                tmp_data = data_rest_part+data_new_part
                tmp_labels = labels_rest_part+labels_new_part
                return tmp_data, tmp_labels
            else:
                self._index_in_epoch = 0
                return data_rest_part, labels_rest_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]