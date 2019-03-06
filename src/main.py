#!/usr/bin/python3

from dataset.mnist import *
from model.network import Network
from task_a import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import time
import calendar
import logging

from pprint import pprint

if __name__ == '__main__':

    # get timestamp
    ts = calendar.timegm(time.gmtime())

    if not os.path.exists("tmp/"):
        os.makedirs("tmp/")
    # init logging
    logging.basicConfig(
        filename="tmp/logs_" + str(ts) + ".log",
        # filename="tmp/task_a.log",
        filemode='w',
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )

    # tensorflow saver
    saver = tf.train.Saver()

    # Parameters
    batch_size = 128

    learning_rate_a = 0.001
    training_iters_a = 20000
    batch_size_a = 128
    batch_size_fisher_matrix = 1
    task_a_model_name = "tmp/checkpoint_model_a.ckpt"

    learning_rate_b = 0.0001
    training_iters_b = 30
    batch_size_b = 128
    lambda_val = 1.# (1./learning_rate_b)


    # get MNIST data
    mnist = load_mnist('dataset/mnist.pkl.gz')
    mnistA = load_mnist('dataset/mnist.pkl.gz', [0, 1, 2, 3, 4, 5, 6, 7, 8])
    mnistB = load_mnist('dataset/mnist.pkl.gz', [9])

    """ TRAINING DATASET """
    # create train datasets
    trainA = tf.data.Dataset.from_tensor_slices(
        mnistA[0]).batch(batch_size_a, False).repeat()

    # create train datasets
    trainB = tf.data.Dataset.from_tensor_slices(
        mnistB[0]).batch(batch_size_b, False).repeat()

    """ TEST DATASET """
    # create test datasets
    test = tf.data.Dataset.from_tensor_slices(
        mnist[1]).batch(batch_size, False)
    testA = tf.data.Dataset.from_tensor_slices(
        mnistA[1]).batch(batch_size_a, False)
    testB = tf.data.Dataset.from_tensor_slices(
        mnistB[1]).batch(batch_size_b, False)

    """ FISHER MATRIX DATASET """
    # create train datasets
    trainAFM = tf.data.Dataset.from_tensor_slices(
        mnistA[0]).batch(batch_size_fisher_matrix, False)

    """ ITERATORS """
    # create general iterator
    iterator = tf.data.Iterator.from_structure(trainA.output_types,
                                               trainA.output_shapes)
    next_feature, next_label = iterator.get_next()

    # make datasets that we can initialize separately, but using the same structure via the common iterator
    iter_test = iterator.make_initializer(test)
    iter_train_a = iterator.make_initializer(trainA)
    iter_test_a = iterator.make_initializer(testA)
    iter_fisher = iterator.make_initializer(trainAFM)
    iter_train_b = iterator.make_initializer(trainB)
    iter_test_b = iterator.make_initializer(testB)

    """ MODEL """
    # Construct model
    nn = Network(next_feature, next_label)

    # optimizer A
    optimizer_a = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_a)
    update_a = optimizer_a.minimize(nn.loss)

    # fisher matrix opimizer
    fisher_matrix_optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=0)
    fisher_matrix_gradient_tensors = fisher_matrix_optimizer.compute_gradients(
        nn.loss)
    gradients = {}
    variables = {}
    for key, weight in nn.weights.items():
        gradients[key] = tf.Variable(
            tf.zeros(shape=tf.shape(weight)), name="gradient_" + key)
        variables[key] = tf.Variable(
            tf.zeros(shape=tf.shape(weight)), name="variable_" + key)
    for key, bias in nn.biases.items():
        gradients[key] = tf.Variable(
            tf.zeros(shape=tf.shape(bias)), name="gradient_" + key)
        variables[key] = tf.Variable(
            tf.zeros(shape=tf.shape(bias)), name="variable_" + key)

    """ TF SESSION """
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    sess = tf.Session()
    # Run the initializer
    sess.run(init)

    try:
        # restore session
        saver.restore(sess, task_a_model_name)
    except ValueError:

        logging.info("**********")
        logging.info("* TASK A *")
        logging.info("**********")

        compute_task_a(sess, nn, update_a, iter_train_a, iter_test_a,
                       training_iters_a, fisher_matrix_gradient_tensors)
        
        """ SAVING """
        save_path = saver.save(sess, task_a_model_name)
    

    logging.info("**********")
    logging.info("* TASK B *")
    logging.info("**********")

    # optimizer B
    optimizer_b = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate_b)
    ewc, ewc_print = nn.compute_ewc(
        gradients, variables, lambda_val=lambda_val)
    ewc_print = tf.print("****************new iteration ****************", ewc_print, output_stream="file://tmp/tensor_" + str(ts) + ".log")
    update_b = optimizer_b.minimize(
        tf.add(nn.loss, ewc))

    """ TRAINING """
    logging.info("**************")
    logging.info("* TRAINING B *")
    logging.info("**************")

    # init iterator
    sess.run(iter_train_b)

    for i in range(training_iters_b):
        l, _, acc, ewc_print_result = sess.run([nn.loss, update_b, nn.accuracy, ewc_print])
        logging.info(
            "Step: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))

    """ TESTING """
    logging.info("*************")
    logging.info("* TESTING B *")
    logging.info("*************")

    # init iterator
    sess.run(iter_test_b)
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



    """ TESTING """
    logging.info("*************")
    logging.info("* TESTING A *")
    logging.info("*************")

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



    """ TESTING ALL """
    logging.info("***************")
    logging.info("* TESTING ALL *")
    logging.info("***************")

    # init iterator
    sess.run(iter_test)
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



    sess.close()
