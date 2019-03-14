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
import argparse
import logging
import json

from pprint import pprint


def task(**kwargs):
    pprint(kwargs)
    # get timestamp
    ts = calendar.timegm(time.gmtime())
    start_ts = time.time()

    # init logging
    logging.basicConfig(
        # filename="tmp/logs/logs_" + str(ts) + ".log",
        # filemode='w',
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )
    # reset graph
    tf.reset_default_graph()

    # Config
    checkpoint_dir = "checkpoints"
    previous = kwargs.get('previous', '')
    save = kwargs.get('save', '')

    # Parameters
    classes = kwargs.get('classes')
    learn_rate = kwargs.get('learnrate')
    training_iterations = kwargs.get('iterations')
    batch_size = kwargs.get('batch')
    lam = kwargs.get('lambda', 1.)

    display_steps_train = kwargs.get('display', 100)

    # get mnist
    mnist = load_mnist('mnist.pkl.gz')
    mnist_task = load_mnist('mnist.pkl.gz', classes)

    # create train datasets
    train_task = tf.data.Dataset.from_tensor_slices(
        mnist_task[0]).batch(batch_size, False).repeat()

    # create test datasets
    test = tf.data.Dataset.from_tensor_slices(
        mnist[1]).batch(batch_size, False)
    test_task = tf.data.Dataset.from_tensor_slices(
        mnist_task[1]).batch(batch_size, False)

    # create general iterator
    iterator = tf.data.Iterator.from_structure(train_task.output_types,
                                               train_task.output_shapes)
    next_feature, next_label = iterator.get_next()

    # make datasets that we can initialize separately, but using the same structure via the common iterator
    iter_test = iterator.make_initializer(test)
    iter_train_task = iterator.make_initializer(train_task)
    iter_test_task = iterator.make_initializer(test_task)

    # Construct model
    nn = Network(next_feature, next_label)

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learn_rate)
    if previous is not '':
        nn.compute_ewc(lam=lam)
    update = optimizer.minimize(nn.loss)

    # tf Session
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    sess = tf.Session()
    # Run the initializer
    sess.run(init)

    if previous is not '':
        logging.info("* LOAD SESSION *")
        # load checkpoint
        nn.saver.restore(sess, os.path.join(checkpoint_dir, previous, "checkpoint.ckpt"))

        logging.debug("* GRADIENTS & VARIABLES *")
        logging.debug("* gradients *")
        for key, gradient in nn.gradients.items():
            gradient = sess.run(gradient)
            logging.debug(key)
            # logging.debug(gradient)
            logging.debug("min: " + str(gradient.min()))
            logging.debug("max: " + str(gradient.max()))
            logging.debug("-----")
        logging.info("* variables *")
        for key, variable in nn.variables.items():
            variable = sess.run(variable)
            logging.debug(key)
            # logging.debug(variable)
            logging.debug("min: " + str(variable.min()))
            logging.debug("max: " + str(variable.max()))
            logging.debug("---")

    logging.info("* TRAINING CLASSES *")
    nn.train(sess, update, iter_train_task, training_iterations, display_steps=display_steps_train)

    # if model is not saved, fisher claculation unnecessary
    if save is not '' and save is not None:
        logging.info("* CALC FISHER *")
        nn.compute_fisher(sess, iter_test_task)

    logging.info("* TESTING CLASSES *")
    test_classes_result = nn.test(sess, iter_test_task)

    logging.info("* TESTING COMPLETE *")
    test_result = nn.test(sess, iter_test)

    stats = {
        'timestamp': str(ts),
        'time': str(round((time.time() - start_ts), 2)) + ' sec.',
        'classes': classes,
        'learnrate': learn_rate,
        'training_iterations': training_iterations,
        'batch_size': batch_size,
        'lambda': lam,
        'test': {
            'classes': test_classes_result,
            'complete': test_result
        }
    }

    if save is not '' and save is not None:
        logging.info("* SAVE SESSION *")
        # create checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir + save, exist_ok=True)
        # save session
        nn.saver.save(sess, os.path.join(checkpoint_dir, save, "checkpoint.ckpt"))
        # save stats
        stats['gradients'] = {}
        stats['variables'] = {}
        for key, gradient in nn.gradients.items():
            gradient = sess.run(gradient)
            stats['gradients'][key] = {}
            # stats['gradients'][key]['gradient'] = gradient
            stats['gradients'][key]['min'] = float(gradient.min())
            stats['gradients'][key]['max'] = float(gradient.max())
        for key, variable in nn.variables.items():
            variable = sess.run(variable)
            stats['variables'][key] = {}
            # stats['variables'][key]['variable'] = variable
            stats['variables'][key]['min'] = float(variable.min())
            stats['variables'][key]['max'] = float(variable.max())
        with open(checkpoint_dir + save + "/stats.json", 'w') as fp:
            json.dump(stats, fp)
    
    return stats
    


if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser(description='train ewc task')

    # positional arguments
    parser.add_argument('classes', nargs='*', type=int, help='available classes: 0 - 9')
    parser.add_argument('learnrate', type=float, help='optimizer learning rate')
    parser.add_argument('iterations', type=int, help='training iterations')
    parser.add_argument('batch', type=int, help='batch size of datasets')
    parser.add_argument('previous', type=str, help='checkpoint name of previos task')
    parser.add_argument('lambda', type=float, help='ewc lambda value: importance of the old task')

    # optional arguments
    parser.add_argument('-s', '--save', nargs='?', type=str,
                        help='checkpoint name for saving new model')
    parser.add_argument('-d', '--display', nargs='?', type=int,
                        help='print every x steps training results')

    args = parser.parse_args()
    pprint(args)

    task(**vars(args))
