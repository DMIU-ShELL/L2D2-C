# -*- coding: utf-8 -*-
'''
Author: Saptarshi Nath (cosn2)
Description: code to plot metrics from shell system evaluations. Metrics
    plotted include, shell boost factor (i.e., sbf1, sbf2, and sbf3)
'''
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # Make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

def plot_sbf(shell_dir, single_path, name, sbf3_target):
    cwd = os.getcwd() + '/figs/'
    cwd = cwd + '/' + name + '/'
    try:
        os.makedirs(cwd)
    except FileExistsError:
        # directory already exists
        pass



    # Load baseline data for a single LL agent
    print('Loading single agent data')
    single = parse_tensorboard(path=single_path, scalars=['cl_eval/tcr', 'cl_eval/tp'])
    single_tp = single['cl_eval/tp']
    single_tcr = single['cl_eval/tcr']
    print('Single agent data loaded')

    sbf1 = dict()
    sbf2 = dict()
    shell_tcr_dict = dict()

    # Get ShELL experiment data and compute SBF scores
    for experiment in os.listdir(shell_dir):
        shell_data = parse_tensorboard(path=shell_dir + experiment, scalars=['shell_eval/tcr', 'shell_eval/tp'])

        print('Loading shell data for ' + experiment)
        shell_tp = shell_data['shell_eval/tp']
        shell_tcr = shell_data['shell_eval/tcr']
        shell_tcr_dict[experiment] = shell_tcr

        print('Computing ShELL Boost Factor')
        #print(shell_tp, shell_tcr)
        sbf1[experiment] = (shell_tp['value'] / single_tp['value'])
        sbf2[experiment] = (shell_tcr['value'] / single_tcr['value'])

    # sbf1
    f1 = plt.figure(1)
    for experiment in sbf1:
        y = sbf1[experiment]
        y_hat = y.to_numpy().clip(min=0)

        plt.plot(np.arange(0, len(y_hat)), y_hat, label=experiment)

    plt.legend(loc="upper right")
    plt.xlabel("Step")
    plt.ylabel("ShELL Boost Factor 1")
    plt.title("SBF1")
    plt.savefig(cwd + 'SBF1.png')

    # sbf2
    f2 = plt.figure(2)
    for experiment in sbf2:
        y = sbf2[experiment]
        y_hat = y.to_numpy().clip(min=0)

        plt.plot(np.arange(0, len(y_hat)), y_hat, label=experiment)

    plt.legend(loc="upper right")
    plt.xlabel("Step")
    plt.ylabel("Shell Boost Factor 2")
    plt.title("SBF2")
    plt.savefig(cwd + 'SBF2.png')

    # sbf3
    f3 = plt.figure(3)
    for experiment in shell_tcr_dict:
        y = shell_tcr_dict[experiment]['value']
        y_hat = y.to_numpy().clip(min=0)

        plt.plot(np.arange(0, len(y_hat)), y_hat, label=experiment)

    y = single_tcr['value']
    y_hat = y.to_numpy().clip(min=0)

    plt.plot(np.arange(0, len(y_hat)), y_hat, label="Single LL Agent")
    plt.xlabel("Step")
    plt.ylabel("Total Cumulative Reward (TCR)")
    plt.title("SBF3")
    plt.legend(loc="lower right")
    plt.savefig(cwd + 'SBF3.png')


if __name__ == '__main__':
    # Example calls
    # Best to run each one seperately

    parser = argparse.ArgumentParser()
    parser.add_argument('--shell_dir', help='directory of shell experiments')
    parser.add_argument('--single_path', help='path to single agent data to use as baseline')
    parser.add_argument('--name', help='name of folder to save figures to in /figs/')
    parser.add_argument('--sbf3_target', help='target TCR for sbf3 score')
    args = parser.parse_args()

    # Depth 2 CTgraph
    #plot_sbf(shell_dir='/home/lunet/cosn2/DeepRL-0.3/log/CTgraph-v0-shell-depth2/',
    #          single_path='/home/lunet/cosn2/DeepRL-0.3/log/CTgraph-v0-ppo-ss/220615-153935', name='CTgraph Depth 2 Experiments', sbf3_target=)

    # Depth 3 CTgraph
    #plot_sbf(shell_dir='/home/lunet/cosn2/DeepRL-0.3/log/CTgraph-v0-shell-depth3/',
    #          single_path='/home/lunet/cosn2/DeepRL-0.3/log/CTgraph-v0-ppo-ss/220615-154056', name='CTgraph Depth 3 Experiments', sbf3_target=)

    plot_sbf(shell_dir=args.shell_dir, single_path=args.single_path, name=args.name, sbf3_target=args.sbf3_target)
