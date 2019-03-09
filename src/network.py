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

    logits = None
    loss = None
    accuracy = None

    training_gradients = {}
    training_variables = {}

    saver = None

    def __init__(self, x, y):

        with tf.variable_scope("network"):
            self.theta = {
                'wh1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='wh1'),
                'wh2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='wh2'),
                'wh3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3]), name='wh3'),
                'wo': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_classes]), name='wo'),
                'bh1': tf.Variable(tf.random_normal([self.n_hidden_1]), name='bh1'),
                'bh2': tf.Variable(tf.random_normal([self.n_hidden_2]), name='bh2'),
                'bh3': tf.Variable(tf.random_normal([self.n_hidden_3]), name='bh3'),
                'bo': tf.Variable(tf.random_normal([self.n_classes]), name='bo')
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

    def init_fisher_gradients(self):
        # fisher matrix opimizer
        fisher_matrix_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0)
        fisher_matrix_gradient_tensors = fisher_matrix_optimizer.compute_gradients(
            self.loss)

        with tf.variable_scope("training"):
            with tf.variable_scope("gradients"):
                self.training_gradients = {
                    'wh1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='wh1'),
                    'wh2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='wh2'),
                    'wh3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3]), name='wh3'),
                    'wo': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_classes]), name='wo'),
                    'bh1': tf.Variable(tf.random_normal([self.n_hidden_1]), name='bh1'),
                    'bh2': tf.Variable(tf.random_normal([self.n_hidden_2]), name='bh2'),
                    'bh3': tf.Variable(tf.random_normal([self.n_hidden_3]), name='bh3'),
                    'bo': tf.Variable(tf.random_normal([self.n_classes]), name='bo')
                }
            with tf.variable_scope("variables"):
                self.training_variables = {
                    'wh1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='wh1'),
                    'wh2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='wh2'),
                    'wh3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3]), name='wh3'),
                    'wo': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_classes]), name='wo'),
                    'bh1': tf.Variable(tf.random_normal([self.n_hidden_1]), name='bh1'),
                    'bh2': tf.Variable(tf.random_normal([self.n_hidden_2]), name='bh2'),
                    'bh3': tf.Variable(tf.random_normal([self.n_hidden_3]), name='bh3'),
                    'bo': tf.Variable(tf.random_normal([self.n_classes]), name='bo')
                }

        pprint(fisher_matrix_gradient_tensors)
        
        return fisher_matrix_gradient_tensors


    def compute_fisher(self, sess, iterator_initializer, fisher_matrix_gradient_tensors):
        
        """
        probs = tf.nn.softmax(self.logits)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
        """
        
        logging.debug(fisher_matrix_gradient_tensors)

        # init iterator
        sess.run(iterator_initializer)

        iterations = 0
        try:
            while True:
                """
                for grad_name, value in self.theta.items():

                    # compute first-order derivatives
                    grad = sess.run(tf.square(tf.gradients(
                        tf.log(probs[0, class_ind]), value)))

                    result = sess.run(fisher_matrix_gradient_tensors)

                    logging.debug("---------------")
                    logging.debug(grad_name)
                    logging.debug(result)
                    logging.debug(result.min())
                    logging.debug(result.max())

                    # set gradients
                    sess.run(
                        tf.assign(self.training_gradients[grad_name], tf.square(result)))
                """

                for grad in fisher_matrix_gradient_tensors:

                    grad_name = grad[1].name
                    start_index = grad_name.rfind('/') + 1
                    end_index = grad_name.rfind(':')
                    grad_name = grad_name[start_index:end_index]
                    # print(grad_name)

                    if grad[1].name[:start_index] == "":
                        continue
                    if grad[1].name[:start_index] != "network/":
                        continue
                    if grad[0] is None:
                        continue

                    # calc gradients
                    gr, vr = sess.run([
                        tf.assign(self.training_gradients[grad_name], tf.square(grad[0])),
                        tf.assign(self.training_variables[grad_name], grad[1])
                    ])

                    # debug logs
                    # logging.debug("---------------")
                    # logging.debug("gradient")
                    # logging.debug(grad_name)
                    # logging.debug(gr)
                    # logging.debug(gr.min())
                    # logging.debug(gr.max())

                    # logging.debug("variable")
                    # logging.debug(vr)
                    # logging.debug(vr.min())
                    # logging.debug(vr.max())
                    
                iterations += 1
        except tf.errors.OutOfRangeError:

            logging.info("* GRADIENTS & VARIABLES iteration over *")

            for key, grad in self.training_gradients.items():
                div = sess.run(tf.truediv(
                    grad, tf.cast(iterations, tf.float32)))
                self.training_gradients[key] = self.training_gradients[key].assign(div)

                # logging.debug("key: " + key)
                # logging.debug("gradient: " + str(div))
                # logging.debug("shape: " + str(div.shape))
                # logging.debug("min: " + str(div.min()))
                # logging.debug("max: " + str(div.max()))
                # logging.debug("-----")
            
            # for key, variable in self.training_variables.items():
            #     variable = sess.run(variable)
            #     logging.debug("key: " + key)
            #     logging.debug("variable: " + str(variable))
            #     logging.debug("shape: " + str(variable.shape))
            #     logging.debug("min: " + str(variable.min()))
            #     logging.debug("max: " + str(variable.max()))
            #     logging.debug("---")


    def compute_ewc(self, lam):
        
        logging.debug("* CALC EWC *")

        prints = []

        ewc_loss = self.loss
        for key, gradient in self.training_gradients.items():
            
            # calc EWC appendix
            subAB = tf.subtract(self.theta[key], self.training_variables[key])
            powAB = tf.square(subAB)
            multiplyF = tf.multiply(powAB, gradient)

            prints.append("------------------")
            prints.append(key)
            prints.append("subAB")
            prints.append(subAB)
            prints.append("----")
            prints.append("powAB")
            prints.append(powAB)
            prints.append("----")
            prints.append("GRADIENT")
            prints.append("min: ")
            prints.append(tf.reduce_max(gradient))
            prints.append("max: ")
            prints.append(tf.reduce_min(gradient))
            prints.append("complete")
            prints.append(gradient)
            prints.append("----")
            prints.append("multiplyF")
            prints.append(multiplyF)
            prints.append("----")
            prints.append("reduce_sum with lambda")
            prints.append((lam/2) * tf.reduce_sum(multiplyF))
            prints.append("----")
            prints.append("loss without ewc")
            prints.append(self.loss)

            ewc_loss += (lam/2) * tf.reduce_sum(multiplyF)

            prints.append("----")
            prints.append("ewc_loss")
            prints.append(ewc_loss)
        
        prints.append("----------------------------------")
        prints.append("ITERATION LOSS")
        prints.append(ewc_loss)

        return ewc_loss, prints

    def train(self, sess, update, training_iters, iter_init, run=[]):
        # init iterator
        sess.run(iter_init)

        for i in range(training_iters):
            l, _, acc = sess.run([self.loss, update, self.accuracy] + run)
            if i % 100 == 0:
                logging.info(
                    "Step: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))

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
            logging.info("Average validation set accuracy over {} iterations is {:.2f}%".format(
                iterations, (avg_acc / iterations) * 100))
        
        return (iterations, (avg_acc / iterations) * 100)
