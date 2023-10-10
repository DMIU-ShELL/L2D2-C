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
    '''Produces sbf1, 2 and 3 graphs, and computes sbf 3 scores'''

    # Sort out the directories to save figures to
    cwd = os.getcwd() + '/figs/'
    cwd = cwd + '/' + name + '/'
    try:
        os.makedirs(cwd)
    except FileExistsError:
        # directory already exists
        pass

    # Load baseline data for a single LL agent
    print('Loading single agent data. This may take a few minutes')
    single = parse_tensorboard(path=single_path, scalars=['cl_eval/tcr', 'cl_eval/tp'])
    single_tp = single['cl_eval/tp']
    single_tcr = single['cl_eval/tcr']


    # Create some dictionaries to hold the shell data
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

        sbf1[experiment] = (shell_tp['value'] / single_tp['value'])
        sbf2[experiment] = (shell_tcr['value'] / single_tcr['value'])

    # SBF1
    f1 = plt.figure(1)
    for experiment in sbf1:
        y = sbf1[experiment]
        y_hat = y.to_numpy().clip(min=0)

        plt.plot(np.arange(0, len(y_hat)), y_hat, label=experiment, alpha=0.5)

    plt.legend(loc="upper right")
    plt.xlabel("Step")
    plt.ylabel("Ratio of TP of single LL agent and ShELL")
    plt.title("ShELL Boost Factor 1")
    plt.savefig(cwd + 'SBF1.png')


    # SBF2
    f2 = plt.figure(2)
    for experiment in sbf2:
        y = sbf2[experiment]
        y_hat = y.to_numpy().clip(min=0)

        plt.plot(np.arange(0, len(y_hat)), y_hat, label=experiment, alpha=0.5)

    plt.legend(loc="upper right")
    plt.xlabel("Step")
    plt.ylabel("Ratio of the time-specific total cumulative reward")
    plt.title("ShELL Boost Factor 2")
    plt.savefig(cwd + 'SBF2.png')


    # SBF3
    f3 = plt.figure(3)

    # SINGLE AGENT EXPERIMENTS PROCESSING

    # Convert wall times to relative times
    wall_time_start = single_tcr['wall_time'].iloc[0]
    single_tcr['wall_time'] = single_tcr['wall_time'].apply(lambda t: (t - wall_time_start) / 3600)

    # Load TCR values and relative times
    y = single_tcr['value']
    x = single_tcr['wall_time']

    # Get wall time (seconds) and compute time elapsed to target TCR
    time_elapsed_single = single_tcr[single_tcr['value'] >= sbf3_target].iloc[0][0]
    y_hat = y.to_numpy().clip(min=0)
    plt.plot(x.to_numpy(), y_hat, label="Single LL Agent", alpha=0.5)


    # SHELL EXPERIMENTS PROCESSING
    for experiment in shell_tcr_dict:
        # Convert wall times to relative times
        wall_time_start = shell_tcr_dict[experiment]['wall_time'].iloc[0]
        shell_tcr_dict[experiment]['wall_time'] = shell_tcr_dict[experiment]['wall_time'].apply(lambda t: (t - wall_time_start) / 3600)

        # Load TCR values and wall times
        y = shell_tcr_dict[experiment]['value']
        x = shell_tcr_dict[experiment]['wall_time']

        # Get wall time (seconds) and compute time elapsed to target TCR
        time_elapsed_shell = shell_tcr_dict[experiment][shell_tcr_dict[experiment]['value'] >= sbf3_target].iloc[0][0]
        y_hat = y.to_numpy().clip(min=0)
        
        sbf3_score = str(time_elapsed_single / time_elapsed_shell)
        plt.plot(x.to_numpy(), y_hat, label=experiment + ': ' + sbf3_score, alpha=0.5)
        print('SBF3 Score for ' + experiment + ': ' + sbf3_score)

    # Add labels, title and legend
    plt.xlabel("Relative elapsed training time (Hours)")
    plt.ylabel("Total Cumulative Reward (TCR)")
    plt.title("Shell Boost Factor 3")
    plt.legend(loc="lower right")
    f3.set_figwidth(15)
    plt.ylim(0,None)
    plt.savefig(cwd + 'SBF3.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shell_dir', help='directory of shell experiments')
    parser.add_argument('--single_path', help='path to single agent data to use as baseline')
    parser.add_argument('--name', help='name of folder to save figures to in /figs/')
    parser.add_argument('--sbf3_target', help='target TCR for sbf3 score')
    args = parser.parse_args()

    #plot_sbf(shell_dir=args.shell_dir, single_path=args.single_path, name=args.name, sbf3_target=args.sbf3_target)

    
    # *Note* Before selecting a SBF3 TCR target, check your tensorboard plots. Selecting a TCR target too high or low will result in a failure to plot

    # Example calls
    # Best to run each one seperately

    # Depth 2 CTgraph
    #plot_sbf(shell_dir='/home/lunet/cosn2/DeepRL-0.3/log/CTgraph-v0-shell-depth2/',
    #          single_path='/home/lunet/cosn2/DeepRL-0.3/log/CTgraph-v0-ppo-ss/220615-153935', name='CTgraph Depth 2 Experiments', sbf3_target=4)

    # Depth 3 CTgraph
    #plot_sbf(shell_dir='/home/lunet/cosn2/DeepRL-0.3/log/CTgraph-v0-shell-depth3/',
    #          single_path='/home/lunet/cosn2/DeepRL-0.3/log/CTgraph-v0-ppo-ss/220615-154056', name='CTgraph Depth 3 Experiments', sbf3_target=6)

    # MiniGrid Simple Crossing
    plot_sbf(shell_dir='/home/lunet/cosn2/DeepRL-0.3/log/MiniGrid-shell-sc_3/',
              single_path='/home/lunet/cosn2/DeepRL-0.3/log/MiniGrid-ppo-ss/220619-010619', name='MiniGrid SC 3 Experiments', sbf3_target=2)

    # MiniGrid Simple Crossing + Lava Crossing
    #plot_sbf(shell_dir='/home/lunet/cosn2/DeepRL-0.3/log/MiniGrid-shell_sc_lc_6/',
    #          single_path='/home/lunet/cosn2/DeepRL-0.3/log/MiniGrid-ppo-ss-sc_lc_6/220619-161455', name='MiniGrid SC LC 6 Experiments', sbf3_target=4)
