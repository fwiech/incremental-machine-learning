#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import sys
import os
import calendar
import argparse
import json

from pprint import pprint

def plotter(taskA, taskB, title):
    title_font_size = 25
    legend_font_size = 15

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

    complete = dataA['plots']['plots']['Complete']

    lastAindex = dataA['plots']['iters'][-1]

    if 'Inverse_Task' in dataA['plots']['plots']:
        if dataA['plots']['plots']['Inverse_Task'][-1] == 0.:
            inverse_b_iters = [(x+lastAindex) for x in list(dataB['plots']['iters'])]
            task_b = dataB['plots']['plots']['Task']
        else:
            inverse_b_iters = iters + [(x+lastAindex)
                               for x in list(dataB['plots']['iters'])]
            task_b = dataA['plots']['plots']['Inverse_Task'] + dataB['plots']['plots']['Task']
    else:
        inverse_b_iters = [(x+lastAindex)
                           for x in list(dataB['plots']['iters'])]
        task_b = dataB['plots']['plots']['Task']

    task += dataB['plots']['plots']['Inverse_Task']
    iters += [(x+lastAindex) for x in list(dataB['plots']['iters'])]

    complete += dataB['plots']['plots']['Complete']

    plt.plot(iters, task, '^', label="D1", linewidth=4.0,
             markersize=10, markerfacecolor='w', markeredgecolor='black')

    plt.plot(inverse_b_iters, task_b, 'o', label="D2",
             linewidth=4.0, markersize=10, markerfacecolor='w', markeredgecolor='black')

    plt.plot(iters, complete, '-', color='black', label="D1 + D2", linewidth=4.0, markersize=10)

    plt.legend(loc=4, prop=legend_font)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')

    plt.title(title, fontsize=title_font_size)

    plt.show()


if __name__ == '__main__':

    # title = "D9-1 FIM"
    # taskA = "checkpoints91/91_A_FIM/stats.json"
    # taskB = "checkpoints91/91_B_FIM/stats.json"
    # plotter(taskA, taskB, title)

    title = "D9-1"
    taskA = "checkpoints91/91_A_BS1000/stats.json"
    taskB = "checkpoints91/91_B_BS1000/stats.json"
    plotter(taskA, taskB, title)



    # title = "D5-5 FIM"
    # taskA = "checkpoints55/55_A_FIM/stats.json"
    # taskB = "checkpoints55/55_B_FIM/stats.json"
    # plotter(taskA, taskB, title)

    title = "D5-5"
    taskA = "checkpoints55/55_A_BS1000/stats.json"
    taskB = "checkpoints55/55_B_BS1000/stats.json"
    plotter(taskA, taskB, title)



    # title = "P10-10 FIM"
    # taskA = "checkpointsPM/PM_A_FIM/stats.json"
    # taskB = "checkpointsPM/PM_B_FIM/stats.json"
    # plotter(taskA, taskB, title)

    title = "P10-10"
    taskA = "checkpointsPM_IT20k/PM_A_BS1000/stats.json"
    taskB = "checkpointsPM_IT20k/PM_B_BS1000/stats.json"
    plotter(taskA, taskB, title)


    # title = "P10-10 FIM"
    # taskA = "checkpoints/PM_A_FIM/stats.json"
    # taskB = "checkpoints/PM_B_FIM/stats.json"
    # plotter(taskA, taskB, title)

    # title = "P10-10"
    # taskA = "checkpoints/PM_A_BS1000/stats.json"
    # taskB = "checkpoints/PM_B_BS1000/stats.json"
    # plotter(taskA, taskB, title)
