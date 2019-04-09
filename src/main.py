#!/usr/bin/python3

from mnist import *
from network import Network

import tensorflow as tf
import numpy as np

import sys
import os
import time
import calendar
import argparse
import logging
import json


def task(**kwargs):
    # get timestamp
    ts = calendar.timegm(time.gmtime())
    start_ts = time.time()

    # init logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )
    # reset graph
    tf.reset_default_graph()

    # log kwargs
    logging.info(kwargs)

    # Config
    checkpoint_dir = "checkpoints/"
    previous = kwargs.get('previous', '')
    save = kwargs.get('save', '')

    train_network = bool(kwargs.get('train', True))
    logging.info("train network: " + str(train_network))

    # Parameters
    classes = kwargs.get('classes')
    learn_rate = kwargs.get('learnrate')
    training_iterations = kwargs.get('iterations')
    batch_train = kwargs.get('batch_train')

    lam = kwargs.get('lambda', None)
    # check if it is a continual task
    if previous is not '':
        logging.debug("set lambda tp: " + str(lam) + "; " + str(type(lam)))
        if lam is None:
            lam = (1./learn_rate)
        logging.debug("adapted lambda tp" + str(lam) + "; " + str(type(lam)))
    
    permute = kwargs.get('permute', None)

    if save is not '' and save is not None:
        batch_matrix = kwargs.get('batch_matrix')

    display_steps_train = kwargs.get('display', 100)

    # get mnist
    mnist, mnist_task, mnist_task_inverse = load_mnist(classes, permute_seed=permute)

    logging.info(mnist[0][0].shape)
    logging.info(mnist_task[0][0].shape)
    # logging.info(mnist_task_inverse[0][0].shape)

    # create train datasets
    train_task = tf.data.Dataset.from_tensor_slices(
        mnist_task[0]).batch(batch_train, False).repeat()

    if save is not '' and save is not None:
        matrix_task = tf.data.Dataset.from_tensor_slices(
            mnist_task[0]).batch(batch_matrix, False)

    # create test datasets
    test = tf.data.Dataset.from_tensor_slices(
        mnist[1]).batch(batch_train, False)
    test_task = tf.data.Dataset.from_tensor_slices(
        mnist_task[1]).batch(batch_train, False)
    
    if mnist_task_inverse is not None:
        test_task_inverse = tf.data.Dataset.from_tensor_slices(
            mnist_task_inverse[1]).batch(batch_train, False)

    # create general iterator
    iterator = tf.data.Iterator.from_structure(train_task.output_types,
                                               train_task.output_shapes)
    next_feature, next_label = iterator.get_next()

    logging.debug("Iterator:")
    logging.debug(next_feature)
    logging.debug(next_label)

    # make datasets that we can initialize separately, but using the same structure via the common iterator
    iter_test = iterator.make_initializer(test)
    iter_train_task = iterator.make_initializer(train_task)
    iter_test_task = iterator.make_initializer(test_task)
    
    if mnist_task_inverse is not None:
        iter_test_task_inverse = iterator.make_initializer(test_task_inverse)

    if save is not '' and save is not None:
        iter_matrix = iterator.make_initializer(matrix_task)

    # Construct model
    nn = Network(next_feature, next_label, lam)

    # optimizer
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learn_rate
    )
    # check if it is a continual task
    if previous is not '':
        logging.info("* adding EWC penalty term * %f"% (lam,))
        update = optimizer.minimize(
            nn.ewc_loss, var_list=list(nn.theta.values())
        )
    else:
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
        for key in nn.keys:
            gradient = sess.run(nn.gradients[key])
            logging.debug(key)
            logging.debug("min: " + str(gradient.min()))
            logging.debug("max: " + str(gradient.max()))
            logging.debug("-----")
        logging.info("* variables *")
        for key in nn.keys:
            variable = sess.run(nn.variables[key])
            logging.debug(key)
            logging.debug("min: " + str(variable.min()))
            logging.debug("max: " + str(variable.max()))
            logging.debug("---")


    plots = {
        "Task": iter_test_task
    }

    if len(classes) != 10 and previous is not '':
        plots["Complete"] = iter_test
    
    if mnist_task_inverse is not None:
        plots['Inverse_Task'] = iter_test_task_inverse
    
    if train_network:
        logging.info("* TRAINING CLASSES *")
        plots = nn.train(
            sess,
            update,
            iter_train_task,
            training_iterations,
            display_steps_train,
            plots
        )
    else:
        if previous is not '':
            with open(os.path.join(checkpoint_dir, previous, "stats.json")) as json_file:
                data = json.load(json_file)
            plots = data['plots']
        else:
            plots = {}
    

    # if model is not saved, matrix claculation unnecessary
    # if model is loaded, matrix comes from checkpoint so no comp necessary either
    if save is not '' and save is not None:
        logging.info("* CALC MATRIX *")
        nn.compute_matrix(sess, iter_matrix, batch_matrix)

    logging.info("* TESTING ON TRAINED CLASSES *")
    test_classes_result = nn.test(sess, iter_test_task)

    logging.info("* TESTING ON ALL CLASSES *")
    test_result = nn.test(sess, iter_test)

    stats = {
        'timestamp': str(ts),
        'time': str(round((time.time() - start_ts), 2)) + ' sec.',
        'classes': classes,
        'set_sizes': {
            'complete': mnist[0][0].shape[0],
            'task': mnist_task[0][0].shape[0],
        },
        'learnrate': learn_rate,
        'training_iterations': training_iterations,
        'batch_train': batch_train,
        'lambda': lam,
        'test': {
            'classes': test_classes_result,
            'complete': test_result
        },
        'plots': plots
    }
    if permute is not None:
        stats['permute'] = permute

    if save is not '' and save is not None:
        logging.info("* SAVE SESSION *")
        # create checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir + save, exist_ok=True)
        # save session
        save_path = nn.saver.save(sess, os.path.join(checkpoint_dir, save, "checkpoint.ckpt"))
        logging.info(save_path)
        # save stats
        stats['batch_matrix'] = batch_matrix
        stats['gradients'] = {}
        stats['variables'] = {}
        for key, gradient in nn.gradients.items():
            gradient_np = sess.run(gradient)
            stats['gradients'][key] = {
                'min': float(gradient_np.min()),
                'max': float(gradient_np.max())
            }
        for key, variable in nn.variables.items():
            variable_np = sess.run(variable)
            stats['variables'][key] = {
                'min': float(variable_np.min()),
                'max': float(variable_np.max())
            }
        with open(checkpoint_dir + save + "/stats.json", 'w') as fp:
            json.dump(stats, fp)

    return stats



if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser(description='train ewc task')

    # required arguments
    parser.add_argument('--classes', nargs='*', type=int, required=True, help='available classes: 0 - 9')
    parser.add_argument('--learnrate', type=float, required=True, help='optimizer learning rate')
    parser.add_argument('--iterations', type=int, required=True, help='training iterations')

    parser.add_argument('--batch_train', type=int, required=True, help='batch size for training & testing')
    parser.add_argument('--batch_matrix', type=int, required=False, help='batch size for matrix calculation (1=FIM, 2<=GM)')

    # optional arguments
    parser.add_argument('--previous', type=str, required=False,
                        default='', help='checkpoint name of previos task')
    parser.add_argument('-s', '--save', nargs='?', type=str, required=False,
                        default='', help='checkpoint name for saving new model')
    parser.add_argument('-d', '--display', nargs='?', type=int, required=False,
                        default=100, help='print every x steps training results')
    parser.add_argument('--lambda', type=float, required=False, help='optimizer learning rate')
    parser.add_argument('--permute', type=int, required=False, help='permute dataset? If -1: no permutation, if >= 0: random seed for permutation')

    args = parser.parse_args()

    task(**vars(args))
