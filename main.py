#!/usr/bin/python3

from mnist import MNIST
from neural_network import NeuralNetwork

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

if __name__ == '__main__':

    mnist = MNIST()

    print("TRAIN")
    print("Images: ", mnist.train.data.shape)
    print("Labels: ", mnist.train.labels.shape)
    print("TEST")
    print("Images: ", mnist.test.data.shape)
    print("Labels: ", mnist.test.labels.shape)
    print("----")

    # load partials
    train01234 = mnist.get_partial(mnist.train.data, mnist.train.labels, [0,1,2,3,4])
    test01234 = mnist.get_partial(mnist.test.data, mnist.test.labels, [0, 1, 2, 3, 4])
    train5 = mnist.get_partial(mnist.train.data, mnist.train.labels, [5])
    test5 = mnist.get_partial(mnist.test.data, mnist.test.labels, [5])

    print("0-4 TRAIN")
    print("Images: ", train01234.data.shape)
    print("Labels: ", train01234.labels.shape)
    print("0-4 TEST")
    print("Images: ", test01234.data.shape)
    print("Labels: ", test01234.labels.shape)
    print("----")
    print("5 TRAIN")
    print("Images: ", train5.data.shape)
    print("Labels: ", train5.labels.shape)
    print("5 TEST")
    print("Images: ", test5.data.shape)
    print("Labels: ", test5.labels.shape)
    print("----")



    # Parameters
    learning_rate = 0.001
    training_iters_A = 2000
    training_iters_B = 200
    batch_size = 128
    display_step = 100

    # Construct model
    nn = NeuralNetwork(learning_rate=0.0001)


    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    sess = tf.Session()
    # Run the initializer
    sess.run(init)

    nn.train(sess, train01234, 5000, 500)

    print("***********")
    print("* TESTING *")
    print("***********")
    nn.test(sess, {"testset 0-10": mnist.test,
                   "testset 0-4": test01234, "testset 5": test5})


    # compute fisher matrix
    fisher_matrix = nn.compute_fisher_matrix(sess, train01234)

    nn.loss = nn.compute_ewc(gradients=fisher_matrix[0], variables=fisher_matrix[1], lambda_val=1)
    nn.update = nn.optimizer.minimize(nn.loss)

    nn.train(sess, train5, 35, 1, {"zw testset 0-4": test01234})

    print("***********")
    print("* TESTING *")
    print("***********")
    nn.test(sess, {"testset 0-10": mnist.test, "testset 0-4": test01234, "testset 5": test5})


    sess.close()
