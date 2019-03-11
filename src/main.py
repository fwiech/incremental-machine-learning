#!/usr/bin/python3

from dataset.mnist import *
from network import Network

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
        filemode='w',
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )

    checkpoint_name = "checkpoint.ckpt"
    checkpoint_training_a_dir_name = "models/train_a/"
    checkpoint_training_a_fisher_dir_name = "models/train_a_fisher/"

    # Parameters
    batch_size = 128

    learning_rate_a = 0.001
    training_iters_a = 20000
    batch_size_a = 128
    batch_size_fisher_matrix = 1

    learning_rate_b = 0.00001
    training_iters_b = 50
    batch_size_b = 128

    lambda_val = 1200#(1./learning_rate_b)


    # get MNIST data
    mnist = load_mnist('dataset/mnist.pkl.gz')
    mnistA = load_mnist('dataset/mnist.pkl.gz', [0, 1, 2, 3, 4, 5, 6, 7, 8])
    mnistB = load_mnist('dataset/mnist.pkl.gz', [9])

    """ TRAINING DATASET """
    # create train datasets
    trainA = tf.data.Dataset.from_tensor_slices(mnistA[0]).batch(batch_size_a, False).repeat()

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
    trainAFM = tf.data.Dataset.from_tensor_slices((mnistA[0][0], mnistA[0][1])).batch(batch_size_fisher_matrix, False)

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

    fisher_matrix_gradient_tensors = nn.init_fisher_gradients()
    

    """ TF SESSION """
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    sess = tf.Session()
    # Run the initializer
    sess.run(init)



    logging.info("**********")
    logging.info("* TASK A *")
    logging.info("**********")
    
    # check if checkpoint exists
    if os.path.isdir(checkpoint_training_a_dir_name):
        # restore session
        nn.saver.restore(sess, checkpoint_training_a_dir_name + checkpoint_name)

        logging.info("* TESTING A *")
        nn.test(sess, iter_test_a)
    else:

        logging.info("* TRAINING A *")
        nn.train(sess, update_a, training_iters_a, iter_train_a)


        # save session
        os.makedirs(checkpoint_training_a_dir_name)
        save_path = nn.saver.save(
            sess, checkpoint_training_a_dir_name + checkpoint_name)

        logging.info("* TESTING A *")
        nn.test(sess, iter_test_a)



    logging.info("*****************")
    logging.info("* FISHER MATRIX *")
    logging.info("*****************")

    # check if checkpoint exists
    if os.path.isdir(checkpoint_training_a_fisher_dir_name):
        # restore session
        nn.saver.restore(
            sess, checkpoint_training_a_fisher_dir_name + checkpoint_name)
    else:

        nn.compute_fisher(sess, iter_test_a, fisher_matrix_gradient_tensors)
        
        # # save session
        # os.makedirs(checkpoint_training_a_fisher_dir_name)
        # save_path = nn.saver.save(
        #     sess, checkpoint_training_a_fisher_dir_name + checkpoint_name)
        
    # logging
    logging.info("* GRADIENTS & VARIABLES *")
    logging.info("* gradients *")
    for key, gradient in nn.training_gradients.items():
        gradient_np = sess.run(gradient)
        logging.debug(key)
        logging.debug(gradient_np)
        logging.debug("min: " + str(gradient_np.min()))
        logging.debug("max: " + str(gradient_np.max()))
        logging.debug("-----")
    logging.info("* variables *")
    for key, variable in nn.training_variables.items():
        variable_np = sess.run(variable)
        logging.debug(key)
        logging.debug(variable_np)
        logging.debug("min: " + str(variable_np.min()))
        logging.debug("max: " + str(variable_np.max()))
        logging.debug("---")

    # logging.info("fisher done")
    # exit()



    logging.info("**********")
    logging.info("* TASK B *")
    logging.info("**********")


    # optimizer B
    optimizer_b = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate_b)
    ewc, ewc_print = nn.compute_ewc(lam=lambda_val)
    ewc_print = tf.print("****************new iteration ****************",
                         ewc_print, output_stream="file://tmp/tensor_" + str(ts) + ".log")
    update_b = optimizer_b.minimize(ewc)


    logging.info("* TRAINING B *")

    # init iterator
    sess.run(iter_train_b)

    for i in range(training_iters_b):
        l, _, acc, ewc_print_result = sess.run([ewc, update_b, nn.accuracy, ewc_print])
        logging.info(
            "Step: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))



    logging.info("***********")
    logging.info("* TESTING *")
    logging.info("***********")

    logging.info("* TESTING B *")
    nn.test(sess, iter_test_b)

    logging.info("* TESTING A *")
    nn.test(sess, iter_test_a)
    
    logging.info("* TESTING ALL *")
    nn.test(sess, iter_test)


    sess.close()
