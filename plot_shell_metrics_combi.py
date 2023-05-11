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
    ax.set_ylim(0, ylim)
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
    if args_path[-1] != '/':
        args_path += '/'

    names = os.listdir(args_path)
    names = [name for name in names if re.search('agent_*', name) is not None]
    names = sorted(names, key=lambda x: int(x.split('_')[3].split('.')[0]))

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

def load_ll_data(path, interval):
        # Load baseline data for a single LL agent
        if os.path.exists(path + 'eval_metrics_agent_0.csv'):
            path = path + 'eval_metrics_agent_0.csv'
            raw_data = np.loadtxt(path, dtype=np.float32, delimiter=',')
        else:
            path = path + 'eval_metrics.npy'
            raw_data = np.load(path) # shape: num_evals x num_tasks
        # separate wall clock time data
        wall_clock_time = raw_data[ : , -1]
        wall_clock_time = np.expand_dims(wall_clock_time, axis=1)

        raw_data = raw_data[ : , : -1] # remove the last dim (wall clock time)
        #print(raw_data)
        #print(raw_data.shape)
        #raw_data = raw_data[:, 0::interval]
        ll_df = pd.DataFrame(raw_data)
        ll_df = ll_df.groupby(np.arange(len(ll_df))//interval).mean()
        raw_data = ll_df.to_numpy()
        #raw_data = np.array([raw_data])
        #print(raw_data)
        #print(raw_data.shape)

        metrics_icr = raw_data.sum(axis=1)
        metrics_tpot = []
        for idx in range(len(metrics_icr)):
            metrics_tpot.append(sum(metrics_icr[0 : idx]))
        metrics_tpot = np.asarray(metrics_tpot)
        return raw_data, metrics_icr, metrics_tpot, wall_clock_time

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

    maximum_icr_ = 0
    # load single agent (ll) data if it exists
    if args.ll_paths is not None:
        ll_data = []
        ll_icr = []
        ll_tpot = []
        for p in args.ll_paths:
            raw_data, metrics_icr, metrics_tpot, ll_wall_clock = load_ll_data(p, args.interval)
            ll_data.append(raw_data)
            ll_icr.append(metrics_icr)
            ll_tpot.append(metrics_tpot)
        ll_data = np.stack(ll_data, axis=0) # shape: num_seeds x num_evals x num_tasks
        ll_icr = np.stack(ll_icr, axis=0)   # shape: num_seeds x num_evals
        ll_tpot = np.stack(ll_tpot, axis=0) # shape: num_seeds x num_evals
        num_evals = ll_data.shape[1]
        # icr
        data['icr']['LL'] = {}
        data['icr']['LL']['xdata'] = np.arange(num_evals)
        data['icr']['LL']['ydata'] = np.mean(ll_icr, axis=0) # average across seeds
        data['icr']['LL']['ydata_cfi'] = cfi_delta(ll_icr)
        data['icr']['LL']['plot_colour'] = 'red'

        #print(data['icr']['ll']['ydata'])
        maximum_icr_ = np.max(data['icr']['LL']['ydata'])

        #print(data['icr']['ll']['xdata'])
        #print(data['icr']['ll']['ydata'])
        # tpot
        data['tpot']['LL'] = {}
        data['tpot']['LL']['xdata'] = np.arange(num_evals)
        data['tpot']['LL']['ydata'] = np.mean(ll_tpot, axis=0) # average across seeds
        data['tpot']['LL']['ydata_cfi'] = cfi_delta(ll_tpot)
        data['tpot']['LL']['plot_colour'] = 'red'


    '''# Communication Dropout RC
    mypaths = {
        '0% dropout'  : ['log_temp/dropout_experiments_5_RC/baseline/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230307-101747/',
                         'log_temp/dropout_experiments_5_RC/baseline/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230308-174638/',
                         'log_temp/dropout_experiments_5_RC/baseline/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230309-183413/', 
                         'log_temp/dropout_experiments_5_RC/baseline/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230310-114538/',
                         'log_temp/dropout_experiments_5_RC/baseline/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230316-102141/'],

        #'25% dropout' : ['log_temp/dropout_experiments_5_RC/25_dropout_v2/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230308-114202/'],

        '50% dropout' : ['log_temp/dropout_experiments_5_RC/50_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230307-183441/', 
                         'log_temp/dropout_experiments_5_RC/50_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230309-104122/', 
                         'log_temp/dropout_experiments_5_RC/50_dropout/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230309-111351/', 
                         'log_temp/dropout_experiments_5_RC/50_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230310-170655/',
                         'log_temp/dropout_experiments_5_RC/50_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230316-114318/'],

        '75% dropout' : ['log_temp/dropout_experiments_5_RC/75_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230307-194730/', 
                         'log_temp/dropout_experiments_5_RC/75_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230309-100448/', 
                         'log_temp/dropout_experiments_5_RC/75_dropout/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230309-114447/', 
                         'log_temp/dropout_experiments_5_RC/75_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230310-161304/',
                         'log_temp/dropout_experiments_5_RC/75_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230316-121140/'],

        '95% dropout' : ['log_temp/dropout_experiments_5_RC/95_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230308-144104/', 
                         'log_temp/dropout_experiments_5_RC/95_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230309-093817/', 
                         'log_temp/dropout_experiments_5_RC/95_dropout/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230309-133408/', 
                         'log_temp/dropout_experiments_5_RC/95_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230310-150046/',
                         'log_temp/dropout_experiments_5_RC/95_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230320-152646/'],
                         
        '100% dropout': ['log_temp/dropout_experiments_5_RC/100_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230423-105349/',
                         'log_temp/dropout_experiments_5_RC/100_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230423-124640/',
                         'log_temp/dropout_experiments_5_RC/100_dropout/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230423-135325/',
                         'log_temp/dropout_experiments_5_RC/100_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230423-142811/',
                         'log_temp/dropout_experiments_5_RC/100_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230423-145423/']
    }'''


    # ShELL vs LL Comparison RC
    mypaths = {
        '1 agent' : ['log_temp/dropout_experiments_5_RC/LL/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230307-112302/',
                     'log_temp/dropout_experiments_5_RC/LL/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230308-214111/',
                     'log_temp/dropout_experiments_5_RC/LL/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230309-163815/',
                     'log_temp/dropout_experiments_5_RC/LL/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230310-130452/',
                     'log_temp/dropout_experiments_5_RC/LL/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230316-104814/'],

        '4 agents' : ['log_temp/comparison/4_agent/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230308-125222/', 
                     'log_temp/comparison/4_agent/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230310-183612/', 
                     'log_temp/comparison/4_agent/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230310-193733/', 
                     'log_temp/comparison/4_agent/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230310-214659/',
                     'log_temp/comparison/4_agent/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230316-111707/']
                     ,
        '12 agents': ['log_temp/dropout_experiments_5_RC/baseline/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230307-101747/', 
                     'log_temp/dropout_experiments_5_RC/baseline/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230308-174638/', 
                     'log_temp/dropout_experiments_5_RC/baseline/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230309-183413/', 
                     'log_temp/dropout_experiments_5_RC/baseline/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230310-114538/',
                     'log_temp/dropout_experiments_5_RC/baseline/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230316-102141/']
    }


    # ShELL vs LL Comparison RC MINGRID
    #mypaths = {
    #    '4 agent' : ['log/minigrid_comparison_final/4_agent/Minigrid-shell-eval-upz-seed-9157/agent_0/230314-100037/'],
    #    '12 agent': ['log/minigrid_comparison_final/12_agent/Minigrid-shell-eval-upz-seed-9157/agent_0/230314-093336/']
    #}

     # ShELL vs LL Comparison RC MINGRID
    '''mypaths = {
        '1 agent' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_1/Minigrid-shell-eval-upz-seed-9157/agent_0/230314-103103/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_2/Minigrid-shell-eval-upz-seed-9157/agent_0/230405-172155/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_3/Minigrid-shell-eval-upz-seed-9157/agent_0/230406-182007/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_4/Minigrid-shell-eval-upz-seed-9157/agent_0/230407-160010/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_5/Minigrid-shell-eval-upz-seed-9157/agent_0/230408-140338/'],

        '4 agents' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/4_agent/seed_1/Minigrid-shell-eval-upz-seed-9157/agent_0/230314-100037/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/4_agent/seed_2/Minigrid-shell-eval-upz-seed-9157/agent_0/230406-135734/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/4_agent/seed_3/Minigrid-shell-eval-upz-seed-9157/agent_0/230407-093235/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/4_agent/seed_4/Minigrid-shell-eval-upz-seed-9157/agent_0/230407-120057/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/4_agent/seed_5/Minigrid-shell-eval-upz-seed-9157/agent_0/230408-194236/'],
                     
        '12 agents': ['log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_1/Minigrid-shell-eval-upz-seed-9157/agent_0/230314-093336/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_2/Minigrid-shell-eval-upz-seed-9157/agent_0/230406-144616/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_3/Minigrid-shell-eval-upz-seed-9157/agent_0/230406-161710/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_4/Minigrid-shell-eval-upz-seed-9157/agent_0/230407-173448/',
                     'log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_5/Minigrid-shell-eval-upz-seed-9157/agent_0/230407-203908/'],
    }'''

    '''
    mypaths = {
        'seed 1' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_1/Minigrid-shell-eval-upz-seed-9157/agent_0/230314-093336/'],
        'seed 2' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_2/Minigrid-shell-eval-upz-seed-9157/agent_0/230406-144616/'],
        'seed 3' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_3/Minigrid-shell-eval-upz-seed-9157/agent_0/230406-161710/'],
        'seed 4' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_4/Minigrid-shell-eval-upz-seed-9157/agent_0/230407-173448/'],
        'seed 5' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/12_agent/seed_5/Minigrid-shell-eval-upz-seed-9157/agent_0/230407-203908/']
    }
    
    mypaths = {
        'seed 1' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_1/Minigrid-shell-eval-upz-seed-9157/agent_0/230314-103103/'],
        'seed 2' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_2/Minigrid-shell-eval-upz-seed-9157/agent_0/230405-172155/'],
        'seed 3' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_3/Minigrid-shell-eval-upz-seed-9157/agent_0/230406-182007/'],
        'seed 4' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_4/Minigrid-shell-eval-upz-seed-9157/agent_0/230407-160010/'],
        'seed 5' : ['log_temp/minigrid_3_experiments/minigrid_comparison_final/LL/seed_5/Minigrid-shell-eval-upz-seed-9157/agent_0/230408-140338/']
    }


    mypaths = {
        #'global 0% dropout'  : ['log_temp/dropout_experiments_new/0_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230425-235508/'],
        #'local 0% dropout' : ['log_temp/dropout_experiments_new/0_dropout/MetaCTgraph-shell-eval-upz-seed-9170/agent_13/230425-235508/'],
        
        'global 50% dropout' : ['log_temp/dropout_experiments_new/50_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230425-202614/'],
        'local 50% dropout' : ['log_temp/dropout_experiments_new/50_dropout/MetaCTgraph-shell-eval-upz-seed-9170/agent_13/230425-202613/'],

        'global 100% dropout 1' : ['log_temp/dropout_experiments_new/100_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230425-193726/'],                
        'local 100% dropout 1': ['log_temp/dropout_experiments_new/100_dropout/MetaCTgraph-shell-eval-upz-seed-9170/agent_13/230425-193726/'],

        'global 100% dropout 2' : ['log_temp/dropout_experiments_new/100_dropout_run_2/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-140049/'],                
        'local 100% dropout 2': ['log_temp/dropout_experiments_new/100_dropout_run_2/MetaCTgraph-shell-eval-upz-seed-9157/agent_13/230426-140049/'],

        'global 99% dropout' : ['log_temp/dropout_experiments_new/99_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-120459/'],
        'local 99% dropout' : ['log_temp/dropout_experiments_new/99_dropout/MetaCTgraph-shell-eval-upz-seed-9170/agent_13/230426-120459/'],

        'global 95% dropout' : ['log_temp/dropout_experiments_new/95_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-130613/'],
        'local 95% dropout' : ['log_temp/dropout_experiments_new/95_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_13/230426-130613/'],

        'global 0% new interval' : ['log/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-144342/'],
        'local 0% new interval' : ['log/MetaCTgraph-shell-eval-upz-seed-9157/agent_13/230426-144341/']
    }

    mypaths = {
        #'global 0% dropout'  : ['log_temp/dropout_experiments_new/0_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230425-235508/'],
        #'local 0% dropout' : ['log_temp/dropout_experiments_new/0_dropout/MetaCTgraph-shell-eval-upz-seed-9170/agent_13/230425-235508/'],
        
        'global 50% dropout' : ['log_temp/dropout_experiments_new/50_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230425-202614/'],
        'local 50% dropout' : ['log_temp/dropout_experiments_new/50_dropout/MetaCTgraph-shell-eval-upz-seed-9170/agent_13/230425-202613/'],

        'global 100% dropout 1' : ['log_temp/dropout_experiments_new/100_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230425-193726/'],                
        'local 100% dropout 1': ['log_temp/dropout_experiments_new/100_dropout/MetaCTgraph-shell-eval-upz-seed-9170/agent_13/230425-193726/'],

        'global 100% dropout 2' : ['log_temp/dropout_experiments_new/100_dropout_run_2/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-140049/'],                
        'local 100% dropout 2': ['log_temp/dropout_experiments_new/100_dropout_run_2/MetaCTgraph-shell-eval-upz-seed-9157/agent_13/230426-140049/'],

        'global 99% dropout' : ['log_temp/dropout_experiments_new/99_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-120459/'],
        'local 99% dropout' : ['log_temp/dropout_experiments_new/99_dropout/MetaCTgraph-shell-eval-upz-seed-9170/agent_13/230426-120459/'],

        'global 95% dropout' : ['log_temp/dropout_experiments_new/95_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-130613/'],
        'local 95% dropout' : ['log_temp/dropout_experiments_new/95_dropout/MetaCTgraph-shell-eval-upz-seed-9157/agent_13/230426-130613/'],

        #'global 0% new interval' : ['log/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-144342/'],
        #'local 0% new interval' : ['log/MetaCTgraph-shell-eval-upz-seed-9157/agent_13/230426-144341/']

        #'50% global' : ['log/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-171608/'],
        #'50% local seq' : ['log/MetaCTgraph-shell-eval-upz-seed-9157/agent_13/230426-171608/']
    }'''


    # GLOBAL
    mypaths = {
        '0% conn. drop' : ['log_temp/debugging_dropout/0_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-185450/',
                               'log_temp/debugging_dropout/0_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230426-192029/',
                               'log_temp/debugging_dropout/0_dropout/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230426-213356/',
                               'log_temp/debugging_dropout/0_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230426-220237/',
                               'log_temp/debugging_dropout/0_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230428-135111/'],

        '50% conn. drop' : ['log_temp/debugging_dropout/50_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-180954/',
                                'log_temp/debugging_dropout/50_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230426-193836/',
                                'log_temp/debugging_dropout/50_dropout/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230426-210952/',
                                'log_temp/debugging_dropout/50_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230426-225139/',
                                'log_temp/debugging_dropout/50_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230428-000547/'],

        '75% conn. drop' : ['log_temp/debugging_dropout/75_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230428-141246/',
                                'log_temp/debugging_dropout/75_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230428-141246/',
                                'log_temp/debugging_dropout/75_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230428-144111/',
                                'log_temp/debugging_dropout/75_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230428-152224/',
                                'log_temp/debugging_dropout/75_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230428-154110/'],

        '95% conn. drop' : ['log_temp/debugging_dropout/95_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230428-160204/',
                                'log_temp/debugging_dropout/95_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230428-161938/',
                                'log_temp/debugging_dropout/95_dropout/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230428-171501/',
                                'log_temp/debugging_dropout/95_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230429-101237/',
                                'log_temp/debugging_dropout/95_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230429-103542/'],


        '100% conn. drop' : ['log_temp/debugging_dropout/100_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-182409/',
                                 'log_temp/debugging_dropout/100_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230426-201103/',
                                 'log_temp/debugging_dropout/100_dropout/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230426-203653/',
                                 'log_temp/debugging_dropout/100_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230427-232310/',
                                 'log_temp/debugging_dropout/100_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230427-234631/'],
    }


    mypaths = {
        'seed 1' : ['log_temp/debugging_dropout/100_dropout/seed_1/MetaCTgraph-shell-eval-upz-seed-9157/agent_0/230426-182409/'],
        'seed 2' : ['log_temp/debugging_dropout/100_dropout/seed_2/MetaCTgraph-shell-eval-upz-seed-9802/agent_0/230426-201103/'],
        'seed 3' : ['log_temp/debugging_dropout/100_dropout/seed_3/MetaCTgraph-shell-eval-upz-seed-9822/agent_0/230426-203653/'],
        'seed 4' : ['log_temp/debugging_dropout/100_dropout/seed_4/MetaCTgraph-shell-eval-upz-seed-2211/agent_0/230427-232310/'],
        'seed 5' : ['log_temp/debugging_dropout/100_dropout/seed_5/MetaCTgraph-shell-eval-upz-seed-1911/agent_0/230427-234631/']
    }
    



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
        data['icr']['L2D2-C ' + name] = {}
        data['icr']['L2D2-C ' + name]['xdata'] = np.arange(num_evals)
        data['icr']['L2D2-C ' + name]['ydata'] = np.mean(shell_icr, axis=0) # average across seeds
        data['icr']['L2D2-C ' + name]['ydata_cfi'] = cfi_delta(shell_icr)
        data['icr']['L2D2-C ' + name]['plot_colour'] = 'green'

        maximum_icr_ = math.ceil(max(maximum_icr_, np.max(data['icr']['L2D2-C ' + name]['ydata'])))

        # tpot
        data['tpot']['L2D2-C ' + name] = {}
        data['tpot']['L2D2-C ' + name]['xdata'] = np.arange(num_evals)
        data['tpot']['L2D2-C ' + name]['ydata'] = np.mean(shell_tpot, axis=0) # average across seeds
        data['tpot']['L2D2-C ' + name]['ydata_cfi'] = cfi_delta(shell_tpot)
        data['tpot']['L2D2-C ' + name]['plot_colour'] = 'green'



    print(maximum_icr_)
    # plot icr
    fig = plot(data['icr'], 'ICR', yaxis_label='Instant Cumulative Return (ICR)', ylim=maximum_icr_ + 0.5)
    fig.savefig(save_path + 'metrics_icr.pdf', dpi=256, format='pdf', bbox_inches='tight')
    # plot tpot
    fig = plot(data['tpot'], 'TPOT', yaxis_label='Total Performance Over Time (TPOT)', ylim=maximum_icr_ + 0.5)
    fig.savefig(save_path + 'metrics_tpot.pdf', dpi=256, format='pdf', bbox_inches='tight')

    '''
    if args.ll_paths is not None:
        for num_shell_agents in args.num_agents:
            eps = 1e-6 # to help with zero divide
            # tla
            #tla = data['tpot']['shell']['ydata'] / (data['tpot']['ll']['ydata'] + eps)
            tla = (((data['tpot']['shell']['ydata'])[0:len(data['tpot']['ll']['ydata'])] + 0.03125) / ((data['tpot']['ll']['ydata'])[0:len(data['tpot']['shell']['ydata'])] + 0.03125))
            data['tla']['shell'] = {}
            data['tla']['shell']['xdata'] = np.arange(len(tla))#np.arange(num_evals)
            data['tla']['shell']['ydata'] = tla
            data['tla']['shell']['ydata_cfi'] = np.zeros_like(tla)
            data['tla']['shell']['plot_colour'] = 'green'
            # ila
            #ila = data['icr']['shell']['ydata'] / (data['icr']['ll']['ydata'] + eps)
            ila = (((data['icr']['shell']['ydata'])[0:len(data['icr']['ll']['ydata'])] + 0.03125) / ((data['icr']['ll']['ydata'])[0:len(data['icr']['shell']['ydata'])] + 0.03125))
            data['ila']['shell'] = {}
            data['ila']['shell']['xdata'] = np.arange(len(ila))#np.arange(num_evals)
            data['ila']['shell']['ydata'] = ila 
            data['ila']['shell']['ydata_cfi'] = np.zeros_like(ila)
            data['ila']['shell']['plot_colour'] = 'green'
            # plot tla
            y_label = 'TPOT(Shell, t) / TPOT(SingleLLAgent, t)'
            fig = plot(data['tla'], 'Total Learning Advantage (TLA)', yaxis_label=y_label)
            fig.savefig(save_path + 'metrics_tla.pdf', dpi=256, format='pdf', bbox_inches='tight')
            # plot ila
            y_label = 'ICR(Shell, t) / ICR(SingleLLAgent, t)'
            fig = plot(data['ila'], 'Instant Learning Advantage (ILA)', yaxis_label=y_label)
            fig.savefig(save_path + 'metrics_ila.pdf', dpi=256, format='pdf', bbox_inches='tight')
                
            #tra
            # Get the max icr achieved by ll
            max_icr = np.amax(data['icr']['ll']['ydata'])
            # make a nparray from 0 to max icr with a step of 0.1
            icr_steps = np.around(np.arange(0, floor((max_icr+0.1)*10)/10, 0.5), 1)
            #print(icr_steps)
            #icr_steps = np.append(icr_steps, max_icr)
            # Get the index where the icr value is >= i
            # Get the eval_step or time value at that same index
            # Append to single_arr
            single_arr = np.empty(0)
            for i in icr_steps:
                pos = np.where(data['icr']['ll']['ydata'] >= i)
                single_arr = np.append(single_arr, data['icr']['ll']['xdata'][pos[0][0]])
            # Assuming this is with one ShELL experiment at a time.
            # Do same thing for one shell experiment
            shell_arr = np.empty(0)
            for i in icr_steps:
                pos = np.where(data['icr']['shell']['ydata'] >= i)
                shell_arr = np.append(shell_arr, data['icr']['shell']['xdata'][pos[0][0]])

            for i in range(len(single_arr)):
                if single_arr[i] == 0 and shell_arr[i] == 0:
                    shell_arr[i] = eps
                elif single_arr[i] != 0 and shell_arr[i] == 0:
                    single_arr[i] += 1.
                    shell_arr[i] += 1.
            y = np.divide(single_arr, shell_arr)

            # 20% 50% 70%
            y[np.isnan(y)] = 0
            x = np.around(np.arange(0, floor((max_icr+0.1)*10)/10, 0.5, dtype=np.float32), 1)
            #x = np.append(x, max_icr)  # Needed if we want to use >0.1=
            den = max(x)
            for index, val in enumerate(x):
                x[index] = (val/den)*100
                
            NUM_AGENTS = num_shell_agents
            fig = plot_tra(x, y, NUM_AGENTS)

    fig.savefig(save_path + 'TRA.pdf', bbox_inches='tight', \
        dpi=256, format='pdf')'''

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

