#!/usr/bin/python3

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
import csv
import json

from pprint import pprint


if __name__ == '__main__':

    args = {
        'classes': [0,1,2,3,4,5,6,7,8,9],
        'batch': 100,
        'previous': 'PM_A_BS1000',
        # 'lambda': -1,
        'display': 1000,
        'permute': 0,
    }

    rates = [0.001, 0.0001, 0.00001, 0.000001]
    iters = [20000]
    lambdas = [1000, 10000, 100000, 1000000, 1010000, 1050000]
    # rates = [0.00001]
    # iters = range(15, 40, 1)

    with open('analyzePM.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['timestamp', 'duration', 'classes', 'sets', 'learnrate',
                         'iterations', 'batch_size', 'lambda', 'plots', 'test classes', 'test complete'])

        for rate in rates:
            for i in iters:
                for lam in lambdas:
                    args['lambda'] = lam
                    args['learnrate'] = rate
                    args['iterations'] = i
                    result = task(**args)

                    # modify result for stats table
                    result['test_classes'] = str(
                        round(result['test']['classes'], 2)) + "%"
                    result['test_complete'] = str(
                        round(result['test']['complete'], 2)) + "%"
                    result['learnrate'] = str(result['learnrate'])
                    result.pop('test', None)

                    # print stats to table
                    writer.writerow(list(result.values()))

    # write stats
    with open("analyzePM.json", 'w') as fp:
        json.dump(args, fp)
