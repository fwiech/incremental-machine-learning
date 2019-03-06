#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import logging

from pprint import pprint

class Network():

    n_input = 784  # MNIST data input (img shape: 28*28)
    n_hidden_1 = 200  # 1st layer number of neurons
    n_hidden_2 = 200  # 2st layer number of neurons
    n_hidden_3 = 200  # 3st layer number of neurons
    n_classes = 10  # MNIST total classes (0-9 digits)

    weights = {
        'wh1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='wh1'),
        'wh2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='wh2'),
        'wh3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), name='wh3'),
        'wo': tf.Variable(tf.random_normal([n_hidden_3, n_classes]), name='wo')
    }

    biases = {
        'bh1': tf.Variable(tf.random_normal([n_hidden_1]), name='bh1'),
        'bh2': tf.Variable(tf.random_normal([n_hidden_2]), name='bh2'),
        'bh3': tf.Variable(tf.random_normal([n_hidden_3]), name='bh3'),
        'bo': tf.Variable(tf.random_normal([n_classes]), name='bo')
    }

    logits = None
    loss = None
    accuracy = None

    def __init__(self, x, y):
        self.__neural_network__(x, y, self.weights, self.biases)

    def __neural_network__(self, x, y, w: list, b: list):
        # Hidden fully connected layer with 200 neurons
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w['wh1']), b['bh1']))

        # Hidden fully connected layer with 200 neurons
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w['wh2']), b['bh2']))

        # Hidden fully connected layer with 200 neurons
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, w['wh3']), b['bh3']))

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_3, w['wo']) + b['bo']

        # Construct model
        self.logits = out_layer

        # Define loss and optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=y))

        # Evaluate model
        correct_pred = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def compute_ewc(self, gradients, variables, lambda_val):
        
        logging.debug("************")
        logging.debug("* CALC EWC *")
        logging.debug("************")

        prints = []

        ewc = 0
        for key, gradient in gradients.items():

            if key in self.weights:
                varsB = self.weights[key]
            elif key in self.biases:
                varsB = self.biases[key]
            
            # calc EWC appendix
            subAB = tf.subtract(varsB, variables[key])
            powAB = tf.pow(subAB, 2)
            multiplyF = tf.multiply(powAB, gradient)
            lambda_multiply = tf.multiply(multiplyF, lambda_val)

            prints.append("------------------")
            prints.append(key)
            prints.append(subAB)
            prints.append("----")
            prints.append(powAB)
            prints.append("----")
            prints.append(gradient)
            prints.append("----")
            prints.append(multiplyF)
            prints.append("----")
            prints.append(lambda_multiply)
            prints.append("----")
            prints.append(tf.reduce_sum(lambda_multiply))
            prints.append("----")

            if ewc is 0:
                ewc = tf.reduce_sum(lambda_multiply)
                pprint(tf.reduce_sum(lambda_multiply))
            else:
                ewc += tf.reduce_sum(lambda_multiply)
            
            prints.append("CURRENT EWC: ")
            prints.append(ewc)

        return ewc, prints
