#!/usr/bin/python3

from dataset.mnist import *
from network import Network
from main import *

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
import csv

from pprint import pprint


def analyze(**kwargs):
    ts = calendar.timegm(time.gmtime())
    start_ts = time.time()
    
    lam_min = kwargs.get('lam_min')
    lam_max = kwargs.get('lam_max')
    lam_steps = kwargs.get('lam_steps')

    batch_min = kwargs.get('batch_min')
    batch_max = kwargs.get('batch_max')
    batch_steps = kwargs.get('batch_steps')

    learnrates = kwargs.get('learnrates')

    classes = kwargs.get('classes')
    previous = kwargs.get('previous', '')
    training_iterations = kwargs.get('iterations')
    lam = kwargs.get('lambda', 1.)

    data = {
        'timestamp': str(ts),
        'classes': classes,
        'iterations': training_iterations,
        'previous': previous,
    }

    with open('analyze.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['timestamp', 'duration', 'classes', 'learnrate', 'iterations', 'batch_size', 'lambda', 'test classes', 'test complete'])

        for rate in learnrates:
            for lam in range(lam_min, lam_max, lam_steps):
                for batch in range(batch_min, batch_max, batch_steps):
                    # create args dict
                    data['learnrate'] = rate
                    data['lambda'] = lam
                    data['batch'] = batch

                    # train
                    result = task(**data)

                    # modify result for stats table
                    result['test_classes'] = str(round(result['test']['classes'], 2)) + "%"
                    result['test_complete'] = str(round(result['test']['complete'], 2)) + "%"
                    result['learnrate'] = str(result['learnrate'])
                    result.pop('test', None)

                    # print stats to table
                    writer.writerow(list(result.values()))

    # append stats
    data['batch'] = {
        'min': batch_min,
        'max': batch_max,
        'step': batch_steps
    }
    data['lambda'] = {
        'min': lam_min,
        'max': lam_max,
        'step': lam_steps
    }
    data['learnrate'] = learnrates
    data['time'] = str(round((time.time() - start_ts), 2)) + ' sec.'

    # write stats
    with open("analyze.json", 'w') as fp:
        json.dump(data, fp)


if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser(description='analyse ewc values')

    # positional arguments
    parser.add_argument('previous', type=str, help='checkpoint name of previos task; if none, put \'\'')
    parser.add_argument('iterations', type=int, help='training iterations')
    parser.add_argument('learnrates', nargs='+', type=float, help='optimizer learning rate')
    parser.add_argument('classes', nargs='+', type=int,
                        help='available classes: 0 - 9')


    args = parser.parse_args()
    pprint(args)
    args = vars(args)

    # lambda range
    args['lam_min'] = int(input("Enter min lambda value (int): "))
    args['lam_max'] = int(input("Enter max lambda value (int): "))
    args['lam_steps'] = int(input("Enter steps for lambda value (int): "))

    # batch sizes
    args['batch_min'] = int(input("Enter min batch size (int): "))
    args['batch_max'] = int(input("Enter max batch size (int): "))
    args['batch_steps'] = int(input("Enter steps for batch size (int): "))

    analyze(**args)
