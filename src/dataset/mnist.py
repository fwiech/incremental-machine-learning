#!/usr/bin/python3

import numpy as np
import gzip
import pickle
from functools import reduce
import os

def load_mnist(filename='mnist.pkl.gz', classes=[], reshape=False):
    current_dir = os.path.dirname(os.path.abspath(filename))
    filepath = os.path.join(current_dir, filename)

    print(filepath)

    with gzip.open(filepath, 'rb') as f:
        ((traind, trainl), (vald, vall), (testd, testl)
            ) = pickle.load(f, encoding='bytes')
        traind = traind.astype("float32")# .reshape(-1, 28, 28)
        trainl = trainl.astype("float32")
        testd = testd.astype("float32")# .reshape(-1, 28, 28)
        testl = testl.astype("float32")
    
    if reshape:
        traind = traind.reshape(-1, 28, 28)
        testd = testd.reshape(-1, 28, 28)
    
    if len(classes) != 0:
        train = get_partial(traind, trainl, classes)
        test = get_partial(testd, testl, classes)
        return train, test

    return (traind, trainl), (testd, testl)

def get_partial(features, labels, classes=[]):
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

    features_partial = features[classMask]
    labels_partial = labels[classMask]

    return (features_partial, labels_partial)
