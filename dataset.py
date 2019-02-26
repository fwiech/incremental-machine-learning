#!/usr/bin/python3

import numpy as np
import logging

class Dataset():

    data = None
    shaped_data = None
    labels = None

    batch_size = 0
    iteration = 0
    epoch = 0

    def __init__(self, data, labels, batch_size=128):
        self.data = data
        self.shaped_data = self.data.reshape(-1, 784)
        self.labels = labels
        self.batch_size = batch_size

    def get_batch(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        if self.iteration >= (self.shaped_data.shape[0] // self.batch_size):
            self.iteration = 0
            self.epoch += 1

        dataBatch = self.shaped_data[self.iteration *
                                        self.batch_size:(self.iteration+1) * self.batch_size]
        labelBatch = self.labels[self.iteration *
                                    self.batch_size:(self.iteration+1) * self.batch_size]
        self.iteration += 1

        return dataBatch, labelBatch, self.iteration, self.epoch

    def reset_batch(self):
        self.iteration = 0
        self.epoch = 0
