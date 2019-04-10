#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import gzip
import pickle
from functools import reduce
import os
from copy import deepcopy

possible_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def load_mnist(classes=[], permute_seed=None, reshape=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # reshape train labels
    y_train_ = np.zeros((y_train.shape[0], 10), dtype=float)
    for i in range(len(y_train_)):
        num = y_train[i]
        y_train_[i][num] = 1
    y_train = y_train_

    # reshape test labels
    y_test_ = np.zeros((y_test.shape[0], 10), dtype=float)
    for i in range(len(y_test_)):
        num = y_test[i]
        y_test_[i][num] = 1
    y_test = y_test_

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    if not reshape:
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

    invert_classes = []
    invert_classes = [x for x in possible_classes if x not in classes]

    # get partials
    if len(classes) != 0 and len(classes) < 10:

        train = get_partial(x_train, y_train, classes)
        test = get_partial(x_test, y_test, classes)

        invert_train = get_partial(x_train, y_train, invert_classes)
        invert_test = get_partial(x_test, y_test, invert_classes)

        # if permute_seed is not None:
        #     x_train, x_test, train, test, invert_train, invert_test = permute(
        #         permute_seed, x_train, x_test, train, test, invert_train, invert_test
        #     )

        mnist = ((x_train, y_train), (x_test, y_test))
        return mnist, (train, test), (invert_train, invert_test)
    else:
        if permute_seed is not None:
            x_train_perm = deepcopy(x_train)
            x_test_perm = deepcopy(x_test)

            x_train_perm, x_test_perm = permute(
                permute_seed, x_train_perm, x_test_perm)

            return ((x_train + x_train_perm, y_train + y_train), (x_test + x_test_perm, y_test + y_test)), ((x_train_perm, y_train), (x_test_perm, y_test)), ((x_train, y_train), (x_test, y_test))

        return ((x_train, y_train), (x_test, y_test)), ((x_train, y_train), (x_test, y_test)), None

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

def permute(seed=None, *args):
    perm = np.arange(0, args[0].shape[1])
    np.random.seed(seed)
    np.random.shuffle(perm)

    for arg in args:
        arg[:, :] = arg[:, perm]

    return args
