#!/usr/bin/python3

from dataset.mnist import *
from model.network import Network

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
import calendar
import logging

from pprint import pprint


def compute_task_a(sess, nn, update, iter_train_a, iter_test_a, iter_fisher, training_iters_a):
    """ TRAINING """
    logging.info("************")
    logging.info("* TRAINING *")
    logging.info("************")

    # init iterator
    sess.run(iter_train_a)

    for i in range(training_iters_a):
        l, _, acc = sess.run([nn.loss, update, nn.accuracy])
        if i % 100 == 0:
            logging.info(
                "Step: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))

    """ TESTING """
    logging.info("***********")
    logging.info("* TESTING *")
    logging.info("***********")

    # init iterator
    sess.run(iter_test_a)
    iterations = 0
    avg_acc = 0
    try:
        while True:
            acc = sess.run([nn.accuracy])
            avg_acc += acc[0]
            iterations += 1
    except tf.errors.OutOfRangeError:
        logging.info("Average validation set accuracy over {} iterations is {:.2f}%".format(
            iterations, (avg_acc / iterations) * 100))

    """ FISHER MATRIX """
    logging.info("*****************")
    logging.info("* FISHER MATRIX *")
    logging.info("*****************")

    nn.compute_fisher(sess, iter_fisher)

    return sess, nn

