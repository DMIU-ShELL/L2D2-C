import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import glob
import argparse
from collections import OrderedDict
from copy import deepcopy
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def cfi_delta(data, conf_int_param=0.95): # confidence interval
    mean = np.mean(data, axis=1)
    if data.ndim == 1:
        std_error_of_mean = st.sem(data, axis=1)
        lb, ub = st.t.interval(conf_int_param, df=len(data)-1, loc=mean, scale=std_error_of_mean)
        cfi_delta = ub - mean
    elif data.ndim == 2:
        std_error_of_mean = st.sem(data, axis=1)
        #print(std_error_of_mean)
        lb, ub = st.t.interval(conf_int_param, df=data.shape[0]-1, loc=mean, scale=std_error_of_mean)
        cfi_delta = ub - mean
        cfi_delta[np.isnan(cfi_delta)] = 0.
    else:
        raise ValueError('`data` with > 2 dim not expected. Expect either a 1 or 2 dimensional tensor.')
    return cfi_delta

def plot(master, title='', xaxis_label='Iterations', yaxis_label='Return'):
    #fig = plt.figure(figsize=(25, 6))  # For wide graph
    fig = plt.figure(figsize=(30, 6))
    ax = fig.subplots()

    ax.set_xlabel(xaxis_label)
    ax.xaxis.label.set_fontsize(20) # Originally 30
    ax.set_ylabel(yaxis_label)
    ax.yaxis.label.set_fontsize(20) # Originally 30
    ax.set_ylim(-0.1, 1.0)
    # axis ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(axis='both', which='major', labelsize=20)
    # remove right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set left and bottom spines at (0, 0) co-ordinate
    ax.spines['left'].set_position(('data', 0.0))
    ax.spines['right'].set_position(('data', 0.0))
    # draw dark line at the (0, 0) co-ordinate
    ax.axhline(y=-0.1, color='k')
    ax.axvline(x=0, color='k')
    # set grid lines
    ax.grid(True, which='both')
        
    for method_name, result_dict in master.items():
        
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        plot_colour = result_dict['plot_colour']
        ax.plot(xdata, ydata, linewidth=3, label=method_name, alpha=0.5)
        ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2)
    # legend
    ax.legend(loc='lower right', prop={'size': 15}, bbox_to_anchor=(1.05, 0.0))
    return fig

def plot_sum(fig, ax, master, title='', xaxis_label='Iteration', yaxis_label='Summed Return'):
    """
    This function creates a visualization of the cumulative return (sum of average returns) 
    across iterations for multiple experiments in the provided data structure.

    Args:
        master (dict): A dictionary containing data for each experiment.
            - Key: Name of the experiment (method_name)
            - Value: A dictionary containing:
                - xdata (np.array): Iteration numbers.
                - ydata (np.array): Average return across seed runs for each iteration.
                - ydata_cfi (np.array): Confidence interval for the average return at each iteration.
                - plot_colour (str): Color to be used for plotting the experiment's results.
        title (str, optional): Title for the plot (defaults to '').
        xaxis_label (str, optional): Label for the x-axis (defaults to 'Iteration').
        yaxis_label (str, optional): Label for the y-axis (defaults to 'Cumulative Return').
    """

    # Initialize cumulative return (zeros for the same shape as one experiment's ydata)
    cumulative_return = np.zeros_like(master[list(master.keys())[0]]['ydata'])
    cumulative_cfi = np.zeros_like(cumulative_return)

    # Loop through experiments and plot individual lines with confidence interval fill
    for method_name, result_dict in master.items():
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        plot_colour = result_dict['plot_colour']

        #ax.plot(xdata, ydata, linewidth=3, label=method_name, alpha=0.5)
        #ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2, color=plot_colour)

        # Add current experiment's average return to cumulative return
        cumulative_return += result_dict['ydata']
        # Add current experiment's CFI to cumulative CFI (element-wise)
        cumulative_cfi += result_dict['ydata_cfi']

    # Plot the cumulative return line
    ax.plot(xdata, cumulative_return, linewidth=3, label=title)  # Adjust color as needed
    ax.fill_between(xdata, cumulative_return - cumulative_cfi/2, cumulative_return + cumulative_cfi/2, alpha=0.2)  # Adjust color as needed

    # Legend
    ax.legend(loc='lower right', prop={'size': 15})

    return fig, ax

def plot_box(fig, ax, master, title='', xaxis_label='Iteration', yaxis_label='Summed Return'):
    return

def assess_policy_stability(rewards, window_size=10, threshold_ratio=0.95):
    """
    This function assesses the stability of an RL policy based on moving average reward and a threshold ratio of the maximum reward.

    Args:
        rewards: A list of rewards obtained during training (length 199 in your case).
        window_size: The window size for calculating the moving average reward.
        threshold_ratio: The threshold ratio of the maximum observed reward for assessing stability.

    Returns:
        A dictionary containing:
            - stable_timestep: The timestep at which the policy is considered stable (None if not found).
            - sample_efficiency: None (not calculated in this version).
    """
    results = {"stable_timestep": None}

    # Calculate maximum reward (assuming all rewards are positive)
    max_reward = np.max(rewards)

    # Calculate moving average reward
    moving_average_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    # Identify potential stable periods
    stable_periods = []
    threshold_reward = max_reward * threshold_ratio  # Dynamic threshold based on max_reward
    for i in range(len(moving_average_rewards) - 1):
        if moving_average_rewards[i] >= threshold_reward and moving_average_rewards[i + 1] >= threshold_reward:
            stable_periods.append((i, i + 1))

    # Check if any stable periods exist
    if not stable_periods:
        return results

    # Identify the most recent stable period with the longest duration
    longest_stable_period = max(stable_periods, key=lambda p: p[1] - p[0])
    results["stable_timestep"] = longest_stable_period[0]

    return results

def plot_tra(master, title='', xaxis_label='', yaxis_label=''):
    results = {}
    names = []
    for experiment_name, experiment_data in master.items():
        results[experiment_name] = {}
        for name, data, in experiment_data.items():
            max_reward = np.amax(data['ydata'])

            rewards = data['ydata']
            for i, reward in enumerate(rewards):
                if reward >= 0.8 * max_reward:
                    max_index = data['xdata'][i]
                    break
            #print(experiment_name, name, max_index)

            if max_index == 0: max_index = np.amax(data['xdata'])

            names.append(name)
            results[experiment_name][name] = {
                "max_y" : max_reward,
                "x_at_max_y" : max_index,
                "max_index_diff" : None
            }

    x_data = []
    y_data = []

    names = list(OrderedDict.fromkeys(names))
    #names = list(sorted(set(names)))
    experiments = list(results.keys())

    #print(names)

    for name in names:
        for exp1 in experiments:
            if exp1 == 'Isolated agents':
                for exp2 in experiments:
                    if exp2 == 'C3L':
                        value1 = results[exp1][name]["x_at_max_y"]
                        value2 = results[exp2][name]["x_at_max_y"]
                        diff = value1 - value2
                        if diff >= 0:
                            #print(exp1, exp2, name, diff, value1, value2)
                            results[exp1][name]["max_index_diff"] = diff
                            y_data.append(diff)
                            x_data.append(name)

    #print(x_data)
    #print(y_data)

    fig = plt.figure(figsize=(30, 6))
    ax = fig.subplots()
    
    ax.bar(x_data, y_data)

    # Calculate the y-axis offset for text placement (adjust as needed)
    y_offset = 0.1

    # Loop through data and add text annotations above each bar
    for i, value in enumerate(y_data):
        ax.text(x_data[i], value + y_offset, str(value), ha='center', va='bottom', fontsize=12)  # Adjust ha, va, and fontsize as needed
    

    ax.set_xlabel("Task")
    ax.set_ylabel("Time Reduction Advantage (TRA)")
    ax.tick_params(axis='x', rotation=90)
    fig.savefig(f'./log/plots/tra.pdf', dpi=256, format='pdf', bbox_inches='tight')

# MCTGRAPH cumulative only
mypaths1 = {
    ############## MCTGRAPH
    # fullcomm
    'C3L' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T19/',
    },

    # nocomm
    'Isolated agents' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T19/',
    }
}

# MINIGRID cumulative and individual graphs
mypaths2 = {
    ############## MINIGRID
    # fullcomm
    'C3L' : {
        'SC1' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T0/',
        'LC1' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T1/',
        'SC2' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T2/',
        'LC2' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T3/',
        'CM2' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T4/',
        'CM3' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T5/',
        'CM4' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T6/',
        'CM5' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T7/',
        'CM6' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T8/',
        'CM7' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T9/',
        'CM8' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T10/',
        'CM9' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T11/',
        'CM10' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T12/',
        'CM11' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T13/'
    },

    # nocomm
    'Isolated agents' : {
        'SC1' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T0/',
        'LC1' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T1/',
        'SC2' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T2/',
        'LC2' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T3/',
        'CM2' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T4/',
        'CM3' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T5/',
        'CM4' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T6/',
        'CM5' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T7/',
        'CM6' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T8/',
        'CM7' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T9/',
        'CM8' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T10/',
        'CM9' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T11/',
        'CM10' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T12/',
        'CM11' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T13/'
    }
}

# PROCGEN cumulative and indvidual graphs
mypaths3 = {
    ############## PROCGEN
    # fullcomm
    'procgen_full_comm' : {
        'Level 0' : 'AAAI_EXPERIMENTS/procgen/full_comm/T0/',
        'Level 1' : 'AAAI_EXPERIMENTS/procgen/full_comm/T1/',
        'Level 2' : 'AAAI_EXPERIMENTS/procgen/full_comm/T2/',
        'Level 3' : 'AAAI_EXPERIMENTS/procgen/full_comm/T3/',
        'Level 4' : 'AAAI_EXPERIMENTS/procgen/full_comm/T4/',
        'Level 5' : 'AAAI_EXPERIMENTS/procgen/full_comm/T5/',
        'Level 6' : 'AAAI_EXPERIMENTS/procgen/full_comm/T6/',
        'Level 7' : 'AAAI_EXPERIMENTS/procgen/full_comm/T7/',
        'Level 8' : 'AAAI_EXPERIMENTS/procgen/full_comm/T8/',
        'Level 9' : 'AAAI_EXPERIMENTS/procgen/full_comm/T9/'
    },

    # nocomm
    'procgen_no_comm' : {
        'Level 0' : 'AAAI_EXPERIMENTS/procgen/no_comm/T0/',
        'Level 1' : 'AAAI_EXPERIMENTS/procgen/no_comm/T1/',
        'Level 2' : 'AAAI_EXPERIMENTS/procgen/no_comm/T2/',
        'Level 3' : 'AAAI_EXPERIMENTS/procgen/no_comm/T3/',
        'Level 4' : 'AAAI_EXPERIMENTS/procgen/no_comm/T4/',
        'Level 5' : 'AAAI_EXPERIMENTS/procgen/no_comm/T5/',
        'Level 6' : 'AAAI_EXPERIMENTS/procgen/no_comm/T6/',
        'Level 7' : 'AAAI_EXPERIMENTS/procgen/no_comm/T7/',
        'Level 8' : 'AAAI_EXPERIMENTS/procgen/no_comm/T8/',
        'Level 9' : 'AAAI_EXPERIMENTS/procgen/no_comm/T9/'
    }
}

# DO NOT USE
mypaths4 = {
    'top_5_masks' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T19/',
    },

    'all_masks_no_simrew' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/all_masks/T19/'
    },

    'reward_condition_only' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/reward_condition/T19/'
    },

    'similarity_only' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/ablations/all_masks/mctgraph/similarity/T19/'
    }
}

# Ablations
mypaths5 = {
    'C3L' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T19/',
    },

    'Similarity only' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_reward_condition/T19/',
    },

    'Reward condition only' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/no_similarity/T19/',
    },

    'All masks' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/ablations/redo/mctgraph/all_masks/T19/',
    },
    
    'Isolated agents' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T6/',
        'Dist 1 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T7/',
        'Dist 1 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T8/',
        'Dist 1 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T9/',
        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T16/',
        'Dist 2 Level 9' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T17/',
        'Dist 2 Level 10' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T18/',
        'Dist 2 Level 11' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T19/',
    },
}

# MCTGRAPH individual graphs
mypaths6 = {
    # fullcomm
    'C3L' : {
        'CL2' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T0/',
        'CL3' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T1/',
        'CL4' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T2/',
        'CL5' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T3/',
        'CL6' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T4/',
        'CL7' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T5/',
        'CL8' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T6/',
        'CL9' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T7/',
        'CL10' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T8/',
        'CL11' : 'AAAI_EXPERIMENTS/mctgraph_combined/full_comm/T9/',
    },

    # nocomm
    'Isolated agents' : {
        'CL2' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T0/',
        'CL3' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T1/',
        'CL4' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T2/',
        'CL5' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T3/',
        'CL6' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T4/',
        'CL7' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T5/',
        'CL8' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T6/',
        'CL9' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T7/',
        'CL10' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T8/',
        'CL11' : 'AAAI_EXPERIMENTS/mctgraph_combined/no_comm/T9/',
    }
}



# MCTGRAPH cumulative only
mypaths7 = {
    'C3L reward adjusted betas (sigmoid)' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/sigmoid_weight_strat/no_top_c/T36/',
    },

    'C3L reward adjusted betas (linear)' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T36/',
    },

    'C3L reward adjusted betas top c (linear)' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/top_c/T36/',
    },
    
    'C3L one reward condition' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T36/',
    },
    
    'C3L two reward condition' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T36/',
    },
    
    'Similarity only' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/sim_only/T36/',
    },

    'Reward condition only' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/rew_only/T36/',
    },

    'All masks' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/all_masks/T36/',
    },
    
    'Isolated agents' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T36/',
    },
}

mypaths8 = {
    'C3L' : {
        'Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T6/',
    },
    
    'Similarity only' : {
        'Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/sim_only/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/sim_only/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/sim_only/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/sim_only/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/sim_only/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/sim_only/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/sim_only/T6/',
    },

    'Reward condition only' : {
        'Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/rew_only/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/rew_only/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/rew_only/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/rew_only/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/rew_only/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/rew_only/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/rew_only/T6/',
    },

    'All masks' : {
        'Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/all_masks/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/all_masks/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/all_masks/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/all_masks/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/all_masks/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/all_masks/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/all_masks/T6/',
    },
    
    'Isolated agents' : {
        'Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T6/',
    },
}

mypaths9 = {
    'C3L one reward condition' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/one_rew_cond/T36/',
    },

    'C3L' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T36/',
    },
    
    'Isolated agents' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T36/',
    },
}

mypaths10 = {
    'C3L' : {
        'Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/full_comm/T6/',
    },
    
    'Isolated agents' : {
        'Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist_combined/no_comm/T6/',
    },
}

# FOUR DIST CTGRAPH SEPERATED
mypaths17 = {
    'C3L' : {
        'Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/full_comm/T6/',
    },
    
    'Isolated agents' : {
        'Level 2' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/ablations/four_dist/no_comm/T6/',
    },
}



# MCTGRAPH ablation cumulative
mypaths11 = {
    'C3L' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T36/',
    }, 

    'No reward mapping' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T36/',
    }, 
    
    'Reward only' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/reward_only/T36/',
    },
    
    'Similarity only' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/similarity_only/T36/',
    },
    
    'All masks' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/all_masks/T36/',
    },

    'Isolated agents' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T36/',
    },
}

# MCTGRAPH ablation seperated
mypaths12 = {
    'C3L' : {
        'Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T6/',
    }, 

    'No reward mapping' : {
        'Level 2' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/beta_parameter_test/c3l/T6/',
    },
    
    'Reward only' : {
        'Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/reward_only/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/reward_only/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/reward_only/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/reward_only/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/reward_only/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/reward_only/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/reward_only/T6/',
    },
    
    'Similarity only' : {
        'Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/similarity_only/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/similarity_only/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/similarity_only/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/similarity_only/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/similarity_only/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/similarity_only/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/similarity_only/T6/',
    },
    
    'All masks' : {
        'Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/all_masks/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/all_masks/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/all_masks/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/all_masks/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/all_masks/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/all_masks/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/all_masks/T6/',
    },

    'Isolated agents' : {
        'Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T6/',
    },
}

# MCTGRAPH ablation cumulative
mypaths13 = {
    'C3L' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T36/',
    },

    'Isolated agents' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T36/',
    },
}

# MCTGRAPH ablation seperated
mypaths14 = {
    'C3L' : {
        'Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_top_c/T6/',
    },

    'Isolated agents' : {
        'Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T0/',
        'Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T1/',
        'Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T2/',
        'Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T3/',
        'Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T4/',
        'Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T5/',
        'Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat_combined/no_comm/T6/',
    },
}

# MINIGRID
mypaths15 = {
    'C3L' : {
        'SC1' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T0/',
        'SC2' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T1/',
        'LC1' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T2/',
        'LC2' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T3/',
        'CM2' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T4/',
        'CM3' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T5/',
        'CM4' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T6/',
        'CM5' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T7/',
        'CM6' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T8/',
        'CM7' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T9/',
        'CM8' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T10/',
        'CM9' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T11/',
        'CM10' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T12/',
        'CM11' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/c3l/T13/',
    },
    
    'Isolated agents' : {
        'SC1' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T0/',
        'SC2' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T1/',
        'LC1' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T2/',
        'LC2' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T3/',
        'CM2' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T4/',
        'CM3' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T5/',
        'CM4' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T6/',
        'CM5' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T7/',
        'CM6' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T8/',
        'CM7' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T9/',
        'CM8' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T10/',
        'CM9' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T11/',
        'CM10' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T12/',
        'CM11' : 'AAAI_EXPERIMENTS/milinear_mapping_minigrid/no_comm/T13/',
    },
}


# MCTGRAPH top_n experiments
mypaths16 = {
    'C3L (no top-n selection)' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_top_c/T36/',
    }, 

    'Top 1 best mask' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_1/T36/',
    },

    'Top 2 best mask' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_2/T36/',
    },

    'Top 5 best mask' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/top_n_experiments/top_5/T36/',
    },

    'Isolated agents' : {
        'Dist 1 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T0/',
        'Dist 1 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T1/',
        'Dist 1 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T2/',
        'Dist 1 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T3/',
        'Dist 1 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T4/',
        'Dist 1 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T5/',
        'Dist 1 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T6/',

        'Dist 2 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T10/',
        'Dist 2 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T11/',
        'Dist 2 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T12/',
        'Dist 2 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T13/',
        'Dist 2 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T14/',
        'Dist 2 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T15/',
        'Dist 2 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T16/',
        
        'Dist 3 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T20/',
        'Dist 3 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T21/',
        'Dist 3 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T22/',
        'Dist 3 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T23/',
        'Dist 3 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T24/',
        'Dist 3 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T25/',
        'Dist 3 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T26/',

        'Dist 4 Level 2' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T30/',
        'Dist 4 Level 3' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T31/',
        'Dist 4 Level 4' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T32/',
        'Dist 4 Level 5' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T33/',
        'Dist 4 Level 6' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T34/',
        'Dist 4 Level 7' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T35/',
        'Dist 4 Level 8' : 'AAAI_EXPERIMENTS/new_weight_strat/no_comm/T36/',
    },
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_name', help='paths to the experiment folder for single'\
        'agent lifelong learning (support paths to multiple seeds)', type=str, default=None)
    parser.add_argument('--exp_name', help='name of experiment', default='metrics_plot')
    parser.add_argument('--num_agents', help='number of agents in the experiment', type=int, nargs='+', default=1)
    parser.add_argument('--interval', help='interval', type=int, default=1)
    args = parser.parse_args()

    MYPATHS = mypaths10
    #MYPATHS = mypaths15
    #MYPATHS = mypaths11

    fig2 = plt.figure(figsize=(30, 6))
    ax2 = fig2.subplots()

    # Set up axis labels, fonts, and limits
    ax2.set_xlabel('Iteration')
    ax2.xaxis.label.set_fontsize(20)
    ax2.set_ylabel('Summed Return')
    ax2.yaxis.label.set_fontsize(20)
    #ax.set_ylim(0.0, 1.0)

    # Axis ticks and grid
    ax2.xaxis.tick_bottom()
    ax2.yaxis.tick_left()
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True, which='both')

    # Remove right and top spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Set left and bottom spines at (0, 0) co-ordinate
    ax2.spines['left'].set_position(('data', 0.0))
    ax2.spines['bottom'].set_position(('data', 0.0))

    # Draw dark lines at (0, 0)
    ax2.axhline(y=0, color='k')
    ax2.axvline(x=0, color='k')

    fig3 = deepcopy(fig2)
    ax3 = deepcopy(ax2)

    master = {}
    master2 = {}

    # Store data for box plot
    boxplot_data = []
    boxplot_labels = []
    #interval_steps = [0, 25, 50, 75, 100, 125, 150, 175, 199]  # Example intervals for box plots
    #interval_steps = [0, 50, 100, 150, 199]
    interval_steps = list(range(0, 200, 1))


    for plot_name, paths in MYPATHS.items():
        print('NAMES:', plot_name, 'PATHS:', paths)
        master2[plot_name] = {}

        for name, path in paths.items():
            print(path)
            data = pd.DataFrame()
            experiment_summed_rewards = []
            for i, filepath in enumerate(sorted(glob.glob(f'{path}*.csv'))):
                # Load data into a pandas dataframe
                df = pd.read_csv(filepath)
                # Select data from second column for each seed run
                data.loc[:, i] = df['Value']
                print(data)

            master[name] = {}
            master[name]['xdata'] = np.arange(data.shape[0])
            master[name]['ydata'] = np.mean(data, axis=1)
            master[name]['ydata_cfi'] = cfi_delta(data)
            master[name]['plot_colour'] = 'green'

            master2[plot_name][name] = {}
            master2[plot_name][name]['xdata'] = np.arange(data.shape[0])
            master2[plot_name][name]['ydata'] = np.mean(data, axis=1)
            master2[plot_name][name]['ydata_cfi'] = cfi_delta(data)
            master2[plot_name][name]['plot_colour'] = 'green'
            
            
            # For each seed, calculate the sum of rewards at the specified interval steps
            for seed in data.columns:
                # Get the rewards for the specific seed
                seed_data = data[seed]

                # Sum the rewards at the specified interval steps for this seed
                values = []
                for step in interval_steps:
                    if step < len(seed_data):
                        #print(seed_data[step])
                        values.append(seed_data[step])
                summed_seed_reward = np.average(values)
                #summed_seed_reward = np.average([seed_data[step] for step in interval_steps if step < len(seed_data)])

                # Append the summed reward for this seed and task to the overall experiment rewards
                experiment_summed_rewards.append(summed_seed_reward)

            # Once all tasks in the experiment are summed, append the data for boxplot
            boxplot_data.extend(experiment_summed_rewards)  # Add all the summed rewards for this experiment
            boxplot_labels.extend([plot_name] * len(experiment_summed_rewards))  # Label with the experiment name

        """
            for step in interval_steps:
                if step < len(master[name]['ydata']):  # Ensure the step is within bounds
                    boxplot_data.append(master[name]['ydata'][step])  # Summed reward at the step
                    boxplot_labels.append(f'{name[0]}{name[-1]}_step_{step}')  # Label for the box plot"""



        if not os.path.exists('./log/plots/'): os.makedirs('./log/plots/')
        fig1 = plot(master, yaxis_label='Return')
        fig1.savefig(f'./log/plots/{plot_name}.pdf', dpi=256, format='pdf', bbox_inches='tight')

        fig2, ax2 = plot_sum(fig2, ax2, master, title=plot_name, yaxis_label='Instant Cumulative Return')

        #fig3, ax3 = plot_box(fig3, ax3, master, title=plot_name, yaxis_label='Summed Return')

    print(len(boxplot_data))
    # Convert boxplot_data to the format required by seaborn
    boxplot_data_df = pd.DataFrame({
        "Average Reward": boxplot_data,
        "Experiment": boxplot_labels
    })

    def remove_duplicates(original_list):
        unique_list = []
        for item in original_list:
            if item not in unique_list:
                unique_list.append(item)
        return unique_list
    
    botplot_ticks = remove_duplicates(boxplot_labels)
    # Create a new figure for box plots
    fig_box, ax_box = plt.subplots(figsize=(8, 6))
    #plt.boxplot(boxplot_data)
    sns.boxplot(x="Experiment", y="Average Reward", data=boxplot_data_df, ax=ax_box)
    ax_box.set_xticklabels(botplot_ticks, rotation=45, ha='right')
    ax_box.set_xlabel('Experiment')
    ax_box.set_ylabel('Reward distrubtion across all tasks')

    # Save figures
    fig_box.savefig('./log/plots/boxplot_comparison.pdf', dpi=256, bbox_inches='tight')  # Save the figure
    fig2.savefig(f'./log/plots/cumulative.pdf', dpi=256, format='pdf', bbox_inches='tight')

    # Plot TRA metric
    plot_tra(master2)