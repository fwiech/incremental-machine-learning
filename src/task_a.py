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


def compute_task_a(sess, nn, update, iter_train_a, iter_test_a, training_iters_a, fisher_matrix_gradient_tensors, gradients, variables):
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

    gradients_np = {}
    variables_np = {}

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

                if gard_name in gradients_np:
                    gradients_np[gard_name] = np.add(
                        gradients_np[gard_name], gradient_power)
                else:
                    gradients_np[gard_name] = gradient_power

                if gard_name not in variables_np:
                    variables_np[gard_name] = result[1]

    except tf.errors.OutOfRangeError:

        logging.debug("*************")
        logging.debug("* gradients_np *")
        for key, gradient in gradients_np.items():
            gradient = np.true_divide(gradient, iterations)
            gradients[key] = gradients[key].assign(gradient.astype(np.float32))
            logging.debug("key: " + key)
            logging.debug("gradient: " + str(gradient))
            logging.debug("shape: " + str(gradient.shape))
            logging.debug("min: " + str(gradient.min()))
            logging.debug("max: " + str(gradient.max()))
            logging.debug("-----")

        logging.debug("*************")
        logging.debug("* VARIABLES *")
        for key, variable in variables_np.items():
            variables[key] = variables[key].assign(variable.astype(np.float32))
            logging.debug("key: " + key)
            logging.debug("variable: " + str(variable))
            logging.debug("shape: " + str(variable.shape))
            logging.debug("min: " + str(variable.min()))
            logging.debug("max: " + str(variable.max()))
            logging.debug("---")

    return gradients, variables
