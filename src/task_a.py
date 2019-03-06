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


def compute_task_a(sess, nn, update, iter_train_a, iter_test_a, training_iters_a, fisher_matrix_gradient_tensors):
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

    # init iterator
    sess.run(iter_test_a)

    gradients = {}
    variables = {}

    iterations = 0
    try:
        while True:
            iterations += 1
            for grad in fisher_matrix_gradient_tensors:
                grad_name = grad[1].name
                gard_name = grad_name.partition(':')[0]

                result = sess.run(grad)
                gradient = result[0]

                gradient_power = np.power(gradient, 2)

                if gard_name in gradients:
                    gradients[gard_name] = np.add(
                        gradients[gard_name], gradient_power)
                else:
                    gradients[gard_name] = gradient_power

                if gard_name not in variables:
                    variables[gard_name] = result[1]

    except tf.errors.OutOfRangeError:

        logging.debug("*************")
        logging.debug("* GRADIENTS *")
        for key, gradient in gradients.items():
            gradient = np.true_divide(gradient, iterations)
            gradients[key] = tf.Variable(gradient, name="gradient_" + key)
            logging.debug("key: " + key)
            logging.debug("shape: " + str(gradient.shape))
            logging.debug("min: " + str(gradient.min()))
            logging.debug("max: " + str(gradient.max()))
            logging.debug("-----")

        logging.debug("*************")
        logging.debug("* VARIABLES *")
        for key, variable in variables.items():
            variables[key] = tf.Variable(variable, name="variable_" + key)
            logging.debug("key: " + key)
            logging.debug("shape: " + str(variable.shape))
            logging.debug("min: " + str(variable.min()))
            logging.debug("max: " + str(variable.max()))
            logging.debug("---")
