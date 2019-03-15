#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import logging

from pprint import pprint

class Network():

    n_input = 784  # MNIST data input (img shape: 28*28)
    n_hidden_1 = 800  # 1st layer number of neurons
    n_hidden_2 = 800  # 2st layer number of neurons
    n_hidden_3 = 800  # 3st layer number of neurons
    n_classes = 10  # MNIST total classes (0-9 digits)

    logits = None
    loss = None
    accuracy = None

    saver = None

    def __init__(self, x, y):

        with tf.variable_scope("network"):
            self.theta = {
                'wh1':  tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='wh1'),
                'wh2':  tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='wh2'),
                'wh3':  tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3]), name='wh3'),
                'wo':   tf.Variable(tf.random_normal([self.n_hidden_3, self.n_classes]), name='wo'),
                'bh1':  tf.Variable(tf.random_normal([self.n_hidden_1]), name='bh1'),
                'bh2':  tf.Variable(tf.random_normal([self.n_hidden_2]), name='bh2'),
                'bh3':  tf.Variable(tf.random_normal([self.n_hidden_3]), name='bh3'),
                'bo':   tf.Variable(tf.random_normal([self.n_classes]), name='bo')
            }
        
        with tf.variable_scope("ewc"):
            with tf.variable_scope("gradients"):
                self.gradients = {
                    'wh1':  tf.Variable(tf.zeros([self.n_input, self.n_hidden_1]), name='wh1', trainable=False),
                    'wh2':  tf.Variable(tf.zeros([self.n_hidden_1, self.n_hidden_2]), name='wh2', trainable=False),
                    'wh3':  tf.Variable(tf.zeros([self.n_hidden_2, self.n_hidden_3]), name='wh3', trainable=False),
                    'wo':   tf.Variable(tf.zeros([self.n_hidden_3, self.n_classes]), name='wo', trainable=False),
                    'bh1':  tf.Variable(tf.zeros([self.n_hidden_1]), name='bh1', trainable=False),
                    'bh2':  tf.Variable(tf.zeros([self.n_hidden_2]), name='bh2', trainable=False),
                    'bh3':  tf.Variable(tf.zeros([self.n_hidden_3]), name='bh3', trainable=False),
                    'bo':   tf.Variable(tf.zeros([self.n_classes]), name='bo', trainable=False)
                }
            with tf.variable_scope("variables"):
                self.variables = {
                    'wh1':  tf.Variable(tf.zeros([self.n_input, self.n_hidden_1]), name='wh1', trainable=False),
                    'wh2':  tf.Variable(tf.zeros([self.n_hidden_1, self.n_hidden_2]), name='wh2', trainable=False),
                    'wh3':  tf.Variable(tf.zeros([self.n_hidden_2, self.n_hidden_3]), name='wh3', trainable=False),
                    'wo':   tf.Variable(tf.zeros([self.n_hidden_3, self.n_classes]), name='wo', trainable=False),
                    'bh1':  tf.Variable(tf.zeros([self.n_hidden_1]), name='bh1', trainable=False),
                    'bh2':  tf.Variable(tf.zeros([self.n_hidden_2]), name='bh2', trainable=False),
                    'bh3':  tf.Variable(tf.zeros([self.n_hidden_3]), name='bh3', trainable=False),
                    'bo':   tf.Variable(tf.zeros([self.n_classes]), name='bo', trainable=False)
                }

        self.__neural_network__(x, y, self.theta)

    def __neural_network__(self, x, y, t: list):
        # Hidden fully connected layer with 200 neurons
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, t['wh1']), t['bh1']))

        # Hidden fully connected layer with 200 neurons
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, t['wh2']), t['bh2']))

        # Hidden fully connected layer with 200 neurons
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, t['wh3']), t['bh3']))

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_3, t['wo']) + t['bo']

        # Construct model
        self.logits = out_layer

        # Define loss and optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=y))

        # Evaluate model
        correct_pred = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # tensorflow saver
        self.saver = tf.train.Saver()

    def compute_fisher(self, sess, iterator_initializer):
        
        # fisher matrix opimizer
        fisher_matrix_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0)
        fisher_matrix_gradient_tensors = fisher_matrix_optimizer.compute_gradients(
            self.loss)

        # log gradient tensor list
        logging.debug(fisher_matrix_gradient_tensors)

        # init iterator
        sess.run(iterator_initializer)

        iterations = 0
        try:
            while True:
                operations = []
                for grad in fisher_matrix_gradient_tensors:

                    grad_name = grad[1].name
                    start_index = grad_name.rfind('/') + 1
                    end_index = grad_name.rfind(':')
                    grad_name = grad_name[start_index:end_index]
                    
                    gradient, variable = sess.run(grad)

                    if iterations is 0:
                        operations.append(
                            tf.assign(self.gradients[grad_name], tf.square(gradient))
                        )
                        operations.append(
                            tf.assign(self.variables[grad_name], variable)
                        )
                    else:
                        operations.append(
                            tf.assign_add(
                                self.gradients[grad_name], tf.square(gradient))
                        )
                
                sess.run(operations)
                iterations += 1
        except tf.errors.OutOfRangeError:

            with tf.variable_scope("gradients"):
                for key, grad in self.gradients.items(): 
                    sess.run(tf.assign(self.gradients[key], tf.truediv(grad, tf.cast(iterations, tf.float32))))

    def compute_ewc(self, lam):
        for key, _ in self.gradients.items():
            # calc EWC appendix
            subAB = tf.subtract(
                self.theta[key], self.variables[key])
            powAB = tf.square(subAB)
            multiplyF = tf.multiply(powAB, self.gradients[key])

            self.loss += (lam/2) * tf.reduce_sum(multiplyF)

    def train(self, sess:tf.Session, update, iter_init, training_iters:int, display_steps=100, *args):
        # init iterator
        sess.run(iter_init)

        for i in range(training_iters):
            l, _, acc, _ = sess.run([self.loss, update, self.accuracy, args])
            if i % display_steps == 0:
                logging.info(
                    "Step: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100.))

    def test(self, sess, iter_init):
        # init iterator
        sess.run(iter_init)

        iterations = 0
        avg_acc = 0
        try:
            while True:
                acc = sess.run([self.accuracy])
                avg_acc += acc[0]
                iterations += 1
        except tf.errors.OutOfRangeError:
            avg_acc = ((avg_acc / float(iterations)) * 100.)
            logging.info("Average validation set accuracy over {} iterations is {:.2f}%".format(iterations, avg_acc))
        
        return avg_acc
