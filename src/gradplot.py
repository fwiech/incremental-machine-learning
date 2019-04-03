#!/usr/bin/python3

from dataset.mnist import *
from network import Network

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import sys
import os
import time
import calendar
import argparse
import logging
import json

from pprint import pprint


def plotter(fim, bs1k, title):
    title_font_size = 25
    legend_font_size = 15

    plt.figure(figsize=(20, 10))

    plt.rcParams.update({'font.size': 25})

    legend_font = fm.FontProperties(size=legend_font_size)

    with open(fim) as json_file:
        fimdata = json.load(json_file)
    with open(bs1k) as json_file:
        bsdata = json.load(json_file)

    fim_grad = fimdata['gradients']
    bs_grad = bsdata['gradients']

    keys = fim_grad.keys()
    keys = tuple(keys)
    key_range = np.arange(len(keys))
    pprint(key_range)

    fim_grad_min = []
    fim_grad_max = []
    bs_grad_min = []
    bs_grad_max = []

    for ob in fim_grad.values():
        fim_grad_min.append(ob['min'])
        fim_grad_max.append(ob['max'])
    
    for ob in bs_grad.values():
        bs_grad_min.append(ob['min'])
        bs_grad_max.append(ob['max'])

    # pprint(fim_grad_min)
    # pprint(bs_grad_min)

    bar_width = 0.35
    opacity = 0.8
    
    # plt.subplot(2, 1, 1)
    plt.subplot()

    plt.bar(key_range, fim_grad_max, bar_width, alpha=opacity,
            color='w', edgecolor='black', hatch='///', label='FIM')

    plt.bar(key_range + bar_width, bs_grad_max, bar_width, alpha=opacity,
            color='w', edgecolor='black', hatch='...', label='BS=1,000')

    plt.xticks(key_range + bar_width, keys)


    plt.legend(loc=1, prop=legend_font)
    plt.ylabel('Maximum Values')
    plt.xlabel('Parameters')
    plt.title(title, fontsize=title_font_size)



    # plt.subplot(2, 1, 2)

    # plt.bar(key_range, fim_grad_min, bar_width,
    #         alpha=opacity, color='w', edgecolor='black', hatch='///', label='FIM')

    # plt.bar(key_range + bar_width, bs_grad_min, bar_width,
    #         alpha=opacity, color='w', edgecolor='black', hatch='...', label='BS=1,000')

    # plt.xticks(key_range + bar_width, keys)


    # plt.legend(loc=1, prop=legend_font)
    # plt.ylabel('min')
    # plt.xlabel('Keys')


    plt.show()


if __name__ == '__main__':


    title = "D9-1"
    fim = "checkpoints91/91_A_FIM/stats.json"
    bs1k = "checkpoints91/91_A_BS1000/stats.json"
    plotter(fim, bs1k, title)

    title = "D5-5"
    fim = "checkpoints55/55_A_FIM/stats.json"
    bs1k = "checkpoints55/55_A_BS1000/stats.json"
    plotter(fim, bs1k, title)

    title = "P10-10"
    fim = "checkpointsPM_IT20k/PM_A_FIM/stats.json"
    bs1k = "checkpointsPM_IT20k/PM_A_BS1000/stats.json"
    plotter(fim, bs1k, title)

