#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import sys
import os
import calendar
import argparse
import json

from pprint import pprint

def plotter(taskA, taskB, title, save=None):
    title_font_size = 25
    legend_font_size = 25

    plt.figure(figsize=(20, 10))

    plt.rcParams.update({'font.size': 25})

    legend_font = fm.FontProperties(size=legend_font_size)

    with open(taskA) as json_file:
        dataA = json.load(json_file)
    with open(taskB) as json_file:
        dataB = json.load(json_file)

    iters = []
    task = []
    complete = []

    iters = dataA['plots']['iters']
    task = dataA['plots']['plots']['Task']

    lastAindex = dataA['plots']['iters'][-1]
    
    task += dataB['plots']['plots']['Inverse_Task']
    iters += [(x+lastAindex) for x in list(dataB['plots']['iters'])]

    inverse_b_iters = [(x+lastAindex)
                       for x in list(dataB['plots']['iters'])]
    task_b = dataB['plots']['plots']['Task']

    complete = dataB['plots']['plots']['Complete']
    complete_iter = inverse_b_iters

    plt.plot(iters, task, '-^', color="blue", label="T1", linewidth=3.0,
             markersize=15, markerfacecolor='w', markeredgecolor='black', markevery=7)

    plt.plot(inverse_b_iters, task_b, '-s', color="red", label="T2",
             linewidth=3.0, markersize=15, markerfacecolor='w', markeredgecolor='black', markevery=7)

    plt.plot(complete_iter, complete, '-o', color='green', label="T1 + T2",
             linewidth=3.0, markersize=15, markerfacecolor='w', markeredgecolor='black', markevery=7)

    plt.axis([0, iters[-1], 0, 100])

    plt.grid()
    plt.margins(0)  # remove default margins (matplotlib verision 2+)
    plt.axvspan(lastAindex, iters[-1], facecolor='black', alpha=0.2)

    plt.legend(loc=4, prop=legend_font)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')

    plt.title(title, fontsize=title_font_size)

    if save is None:
        plt.show()
    else:
        plot_train_dir = os.path.join("plots", "training")
        os.makedirs(plot_train_dir, exist_ok=True)
        plt.savefig(
            os.path.join(plot_train_dir, save + ".png")
        )


if __name__ == '__main__':

        # argparse
    parser = argparse.ArgumentParser(description='plot trained models')

    # required arguments
    parser.add_argument('--title', type=str,
                        required=True, help='plot title')
    parser.add_argument('-t1', '--taskT1', type=str,
                        required=True, help='checkpoint directory path')
    parser.add_argument('-t2', '--taskT2', type=str,
                        required=True, help='checkpoint directory path')

    parser.add_argument('-s', '--save', nargs='?', type=str,
                        required=False, help='plot filename')

    args = parser.parse_args()

    # pprint(vars(args))
    params = vars(args)
    taskA = os.path.join(params['taskT1'], "stats.json")
    taskB = os.path.join(params['taskT2'], "stats.json")
    # print(taskA)
    plotter(taskA, taskB, params['title'], save=params['save'])
