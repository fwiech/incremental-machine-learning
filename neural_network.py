#!/usr/bin/python3

from dataset import Dataset

import tensorflow as tf
import numpy as np
import logging

class NeuralNetwork():

    n_input = 784  # MNIST data input (img shape: 28*28)
    n_hidden_1 = 200  # 1st layer number of neurons
    n_hidden_2 = 200  # 2st layer number of neurons
    n_hidden_3 = 200  # 3st layer number of neurons
    n_classes = 10  # MNIST total classes (0-9 digits)

    # tf Graph
    X = tf.placeholder(tf.float32, [None, n_input], name='x')
    Y = tf.placeholder(tf.float32, [None, n_classes], name='y')

    weights = {
        'wh1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1]), name='wh1'),
        'wh2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]), name='wh2'),
        'wh3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3]), name='wh3'),
        'wo': tf.Variable(tf.truncated_normal([n_hidden_3, n_classes]), name='wo')
    }

    biases = {
        'bh1': tf.Variable(tf.truncated_normal([n_hidden_1]), name='bh1'),
        'bh2': tf.Variable(tf.truncated_normal([n_hidden_2]), name='bh2'),
        'bh3': tf.Variable(tf.truncated_normal([n_hidden_3]), name='bh3'),
        'bo': tf.Variable(tf.truncated_normal([n_classes]), name='bo')
    }

    logits = None
    loss = None
    optimizer = None
    update = None
    accuracy = None

    """
        @log_level:
            * CRITICAL	50
            * ERROR     40
            * WARNING	30
            * INFO	    20
            * DEBUG	    10
            * NOTSET	0
        @filename: write logging to a file. if parameter not set, no file written
    """

    def __init__(self, learning_rate=0.0001, log_level=logging.INFO, log_filename=''):
        
        # init logging
        if log_filename is not '':
            logging.basicConfig(filename=log_filename, level=log_level)
        logging.basicConfig(level=log_level)

        self.__neural_network__(self.X, self.weights,
                                self.biases, learning_rate)
    
    def __neural_network__(self, x: tf.placeholder, w: list, b: list, learning_rate:int):
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
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.update = self.optimizer.minimize(self.loss)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    def train(self, sess: tf.Session, dataset: Dataset, training_iters: int, display_step:int, testsets={}):
        logging.info("************")
        logging.info("* TRAINING *")
        logging.info("************")

        epoch = 0
        for step in range(0, training_iters):
            batch_x, batch_y, _, epoch = dataset.get_batch()

            # Run optimization op (backprop)
            sess.run(self.update, feed_dict={self.X: batch_x, self.Y: batch_y})

            # display output
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                mini_batch_loss, acc = sess.run([self.loss, self.accuracy], feed_dict={
                                                self.X: batch_x, self.Y: batch_y})

                logging.info("Step " + str(step) +
                    ", Epoch" + str(epoch) +
                    ", Minibatch Loss= " +
                    "{:.4f}".format(mini_batch_loss) + ", Training Accuracy= " +
                    "{:.3f}".format(acc))
                
                self.test(sess, testsets)

    def test(self, sess: tf.Session, datasets: {str: Dataset}):
        
        for text, dataset in datasets.items():
            acc = sess.run(self.accuracy, feed_dict={
                           self.X: dataset.shaped_data, self.Y: dataset.labels})
            logging.info(text + ": " + str(acc))

    def compute_fisher_matrix(self, sess: tf.Session, dataset: Dataset):

        zero_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0)
        grads = zero_optimizer.compute_gradients(self.loss)

        gradients = {}
        variables = {}

        epoch_iterations = (dataset.data.shape[0] // dataset.batch_size)
        for _ in range(0, epoch_iterations):
            batch_x, batch_y, _, _ = dataset.get_batch()

            for grad in grads:
                grad_name = grad[1].name
                gard_name = grad_name.partition(':')[0]

                result = sess.run(grad, feed_dict={self.X: batch_x, self.Y: batch_y})
                gradient = result[0]

                gradient_power = np.power(gradient, 2)

                if gard_name in gradients:
                    gradients[gard_name] = np.add(gradients[gard_name], gradient_power)
                else:
                    gradients[gard_name] = gradient_power

                if gard_name not in variables:
                    variables[gard_name] = result[1]

        for key, gradient in gradients.items():
            gradients[key] = np.true_divide(gradient, epoch_iterations)

        return (gradients, variables)

    def compute_ewc(self, gradients: dict, variables: dict, lambda_val=1):

        logging.debug("************")
        logging.debug("* CALC EWC *")
        logging.debug("************")

        # create task A constant
        taskA = {}
        for key, variable in variables.items():
            taskA[key] = tf.constant(variable, name="taskA-" + key)
            logging.debug("key: " + key)
            logging.debug("shape: " + str(variable.shape))
            logging.debug("min: " + str(variable.min()))
            logging.debug("max: " + str(variable.max()))
            logging.debug("---")

        ewc = 0
        for key, gradient in gradients.items():

            # flatten
            varsA = taskA[key]
            gradientA = gradient

            if key in self.weights:
                varsB = self.weights[key]
            elif key in self.biases:
                varsB = self.biases[key]
            
            # calc EWC appendix
            subAB = tf.subtract(varsB, varsA)
            powAB = tf.pow(subAB, 2)
            multiplyF = tf.multiply(powAB, gradientA)
            lambda_multiply = tf.multiply(multiplyF, lambda_val)
            ewc += tf.reduce_sum(lambda_multiply)

        return tf.add(self.loss, ewc)
