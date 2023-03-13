#-*- coding: utf-8 -*-
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import matplotlib.transforms as transforms
import pandas as pd
import scipy.stats as st
import math

def plot(results, title='', xaxis_label='Evaluation checkpoint', yaxis_label='', ylim=16.0):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    # axis title and font
    #ax.set_title(title)
    #ax.title.set_fontsize(22)
    # axis labels and font, and ticks
    ax.set_xlabel(xaxis_label)
    ax.xaxis.label.set_fontsize(20)
    ax.set_ylabel(yaxis_label)
    ax.yaxis.label.set_fontsize(20)
    #ax.set_ylim(0, ylim)
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
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    # set grid lines
    ax.grid(True, which='both')

    #for experiment in experiments:
    #    results = experiment[data_type]

    #    print(len(results))
        
    for method_name, result_dict in results.items():

        print(method_name)
        
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        plot_colour = result_dict['plot_colour']
        ax.plot(xdata, ydata, linewidth=3, label=method_name, alpha=0.5)
        ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2)
    # legend
    ax.legend(loc='lower right')
    return fig

def plot_tra(xdata, ydata, num_agents):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()

    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), \
        ax.transData)

    ax.plot(xdata, ydata, alpha=0.5)
    ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2)
    ax.axhline(y=0.5*int(num_agents), color='red', linestyle='dashed', alpha=0.5)
    ax.text(0.5, 0.4+(0.5*int(num_agents)), "0.5*N="+"{:.0f}".format(0.5*int(num_agents)), \
        color="red", transform=trans, ha="center", va="center", size=14)
    
    #ax.set_title('')
    ax.set_xlabel('Percentage of Target Performance (p)', fontsize=14)
    ax.set_ylabel('TTp(SingleLLAgent) / TTp(L2D2-C)', fontsize=14)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
        
    #addlabels(xdata, ydata)
    return fig

def load_shell_data(args_path, interval):

    paths = [args_path + name for name in names]
    #paths = ['{0}/{1}/'.format(path, os.listdir(path)[-1]) for path in paths]
    num_agents = len(paths)
    metrics = []
    for name, path in zip(names, paths):
        #file_path = path + 'eval_metrics_{0}.csv'.format(name)
        m = np.loadtxt(path, dtype=np.float32, delimiter=',')
        if m.ndim == 1:
            m = np.expand_dims(m, axis=0)
        metrics.append(m)
        #print(m.shape)
    num_evals = metrics[0].shape[0]
    num_tasks = metrics[0].shape[1] - 1 # remove the last dim (wall clock time)
    
    # shape: num_agents x num_evals x num_tasks
    metrics = np.stack(metrics, axis=0)

    # separate wall clock time data
    wall_clock_time = metrics[ : , : , -1]
    wall_clock_time = wall_clock_time.transpose(1, 0)

    metrics = metrics[ : , : , : -1] # remove the last dim (wall clock time) from metrics
    #print(metrics)
    #print(metrics.shape)
    #metrics = metrics[:, :, 0::interval]
    shell_df = pd.DataFrame(metrics[0])
    #shell_df = shell_df.rolling(interval).mean()
    #shell_df = shell_df.iloc[::interval, :]
    shell_df = shell_df.groupby(np.arange(len(shell_df))//interval).mean()
    #print(shell_df_avg)
    metrics = shell_df.to_numpy()
    metrics = np.array([metrics])
    #print(metrics)
    #print(metrics.shape)


    # shape: num_evals x num_agents x num_tasks
    metrics = metrics.transpose(1, 0, 2)
    #print(metrics.shape)
    
    metrics_icr = []
    metrics_tpot = []
    num_evals = len(metrics)
    #print(num_evals)
    for idx in range(num_evals):
        data = metrics[idx]
        _max_reward = data.max(axis=0)
        agent_ids = data.argmax(axis=0).tolist()
        #print('best agent per task: {0}'.format(agent_ids))
        # compute icr/tcr
        icr = _max_reward.sum()
        metrics_icr.append(icr)

        tpot = np.sum(metrics_icr)
        metrics_tpot.append(tpot)
    
    return metrics, metrics_icr, metrics_tpot, wall_clock_time

def cfi_delta(data, conf_int_param=0.95): # confidence interval
    mean = np.mean(data, axis=0)
    if data.ndim == 1:
        std_error_of_mean = st.sem(data, axis=0)
        lb, ub = st.t.interval(conf_int_param, df=len(data)-1, loc=mean, scale=std_error_of_mean)
        cfi_delta = ub - mean
    elif data.ndim == 2:
        std_error_of_mean = st.sem(data, axis=0)
        #print(std_error_of_mean)
        lb, ub = st.t.interval(conf_int_param, df=data.shape[0]-1, loc=mean, scale=std_error_of_mean)
        cfi_delta = ub - mean
        cfi_delta[np.isnan(cfi_delta)] = 0.
    else:
        raise ValueError('`data` with > 2 dim not expected. Expect either a 1 or 2 dimensional tensor.')
    return cfi_delta


def main(args):
    exp_name = args.exp_name
    #save_path = './log/plots/' + os.path.basename(args.path[ : -1]) + '/'
    save_path = './log/plots/' + exp_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = {}
    data['icr'] = {}
    data['tpot'] = {}
    data['tla'] = {}
    data['ila'] = {}
    data['sbf3'] = {}



    # Communication Dropout RC
    mypaths = {
        'DDPPO'  : ['DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_0adbd_00000_0_2023-02-27_15-04-33_WORKERS_2_SEED_959.csv', 'DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_16dcf_00000_0_2023-02-27_17-06-34_WORKERS_2_SEED_960.csv', 'DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_16e7d_00000_0_2023-02-27_14-57-44_WORKERS_2_SEED_958.csv', 'DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_87b2f_00000_0_2023-02-27_23-00-29_WORKERS_2_SEED_962.csv', 'DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_8478f_00000_0_2023-02-27_17-09-38_WORKERS_2_SEED_961.csv'],
        'IMPALA' : ['IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_6f121_00000_0_2023-03-01_18-16-34_WORKERS_2_SEED_959.csv', 'IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_9070f_00000_0_2023-03-01_18-17-30_WORKERS_2_SEED_960.csv', 'IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_9891f_00000_0_2023-03-01_17-27-37_WORKERS_2_SEED_958.csv', 'IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_b7543_00000_0_2023-03-01_18-18-35_WORKERS_2_SEED_961.csv', 'IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_e7696_00000_0_2023-03-01_18-19-55_WORKERS_2_SEED_962.csv'],
        'PPO'    : ['PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_8b766_00000_0_2023-02-27_23-07-45_SEED_962.csv', 'PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_00221_00000_0_2023-03-02_13-49-06_SEED_958.csv', 'PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_cf138_00000_0_2023-02-27_15-10-02_SEED_960.csv', 'PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_d6da2_00000_0_2023-02-27_15-03-06_SEED_959.csv', 'PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_ecece_00000_0_2023-02-27_22-56-09_SEED_961.csv'],
        'L2D2-C' : ['ShELL/2_Agents/Agent1/run-MetaCTgraph-shell-dist-upz-seed-962_agent_1_230310-002012-tag-agent_1_iteration_avg_reward.csv', 'ShELL/2_Agents/Agent2/run-MetaCTgraph-shell-dist-upz-seed-959_agent_2_230308-131206-tag-agent_2_iteration_avg_reward.csv']
    }

    data = pd.DataFrame()
    


    for name, myarr in mypaths.items():
        # load shell data
        shell_data = []
        shell_icr = []
        shell_tpot = []
        for p in myarr:
            raw_data, metrics_icr, metrics_tpot, shell_wall_clock = load_shell_data(p, args.interval)
            shell_data.append(raw_data)
            shell_icr.append(metrics_icr)
            shell_tpot.append(metrics_tpot)

        shell_data = np.stack(shell_data, axis=0) # shape: num_seeds x num_evals x num_agents x num_tasks
        shell_icr = np.stack(shell_icr, axis=0)  # shape: num_seeds x num_evals
        shell_tpot = np.stack(shell_tpot, axis=0) # shape: num_seeds x num_evals
        num_evals = shell_data.shape[1]
        num_shell_agents = args.num_agents#shell_data.shape[2]
        # icr
        data['icr'][name] = {}
        data['icr'][name]['xdata'] = np.arange(num_evals)
        data['icr'][name]['ydata'] = np.mean(shell_icr, axis=0) # average across seeds
        data['icr'][name]['ydata_cfi'] = cfi_delta(shell_icr)
        data['icr'][name]['plot_colour'] = 'green'

        maximum_icr_ = math.ceil(max(maximum_icr_, np.max(data['icr'][name]['ydata'])))

        # tpot
        data['tpot'][name] = {}
        data['tpot'][name]['xdata'] = np.arange(num_evals)
        data['tpot'][name]['ydata'] = np.mean(shell_tpot, axis=0) # average across seeds
        data['tpot'][name]['ydata_cfi'] = cfi_delta(shell_tpot)
        data['tpot'][name]['plot_colour'] = 'green'



    print(maximum_icr_)
    # plot icr
    fig = plot(data['icr'], 'ICR', yaxis_label='Instant Cumulative Return (ICR)', ylim=maximum_icr_+0.5)
    fig.savefig(save_path + 'metrics_icr.pdf', dpi=256, format='pdf', bbox_inches='tight')
    # plot tpot
    fig = plot(data['tpot'], 'TPOT', yaxis_label='Total Performance Over Time (TPOT)', ylim=maximum_icr_+0.5)
    fig.savefig(save_path + 'metrics_tpot.pdf', dpi=256, format='pdf', bbox_inches='tight')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('shell_paths', help='paths to the experiment folder (support'\
    #    'paths to multiple seeds)', nargs='+')
    parser.add_argument('--ll_paths', help='paths to the experiment folder for single'\
        'agent lifelong learning (support paths to multiple seeds)', nargs='+', default=None)
    parser.add_argument('--exp_name', help='name of experiment', default='metrics_plot')
    parser.add_argument('--num_agents', help='number of agents in the experiment', type=int, nargs='+', default=1)
    parser.add_argument('--interval', help='interval', type=int, default=1)
    main(parser.parse_args())

