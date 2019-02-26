#!/usr/bin/python3

from dataset import Dataset

import numpy as np
import gzip
import pickle
from functools import reduce

class MNIST():

    train = None
    test = None

    def __init__(self, dataset_filename='mnist.pkl.gz', batch_size=128):
        train_tuple, test_tuple = self.load_mnist(dataset_filename)

        self.train = Dataset(train_tuple[0], train_tuple[1], batch_size)
        self.test = Dataset(test_tuple[0], test_tuple[1], batch_size)

    def load_mnist(self, dataset_filename='mnist.pkl.gz'):
        with gzip.open(dataset_filename, 'rb') as f:
            ((traind, trainl),(vald, vall), (testd, testl)) = pickle.load(f, encoding='bytes')
            traind = traind.astype("float32").reshape(-1, 28, 28)
            trainl = trainl.astype("float32")
            testd = testd.astype("float32").reshape(-1, 28, 28)
            testl = testl.astype("float32")
        return (traind, trainl), (testd, testl)

    def get_partial(self, data, labels, classes=[]):
        if len(classes) == 0:
            raise ValueError('classes can not be empty')
        
        classVector = labels.argmax(axis=1)
        classMask = None
        for c in classes:
            if not isinstance(c, int):
                raise ValueError('list item must be <int>')
            if classMask is None:
                classMask = (classVector == c)
            else:
                classMask = np.logical_or(classMask, (classVector == c))
        
        data_partial = data[classMask]
        labels_partial = labels[classMask]

        return Dataset(data_partial, labels_partial)
