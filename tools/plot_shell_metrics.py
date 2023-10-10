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

def plot(results, title='', xaxis_label='Evaluation checkpoint', yaxis_label=''):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    # axis title and font
    ax.set_title(title)
    ax.title.set_fontsize(22)
    # axis labels and font, and ticks
    ax.set_xlabel(xaxis_label)
    ax.xaxis.label.set_fontsize(20)
    ax.set_ylabel(yaxis_label)
    ax.yaxis.label.set_fontsize(20)
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

    for method_name, result_dict in results.items():
        print(method_name)
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        plot_colour = result_dict['plot_colour']
        ax.plot(xdata, ydata, linewidth=3, label=method_name, color=plot_colour)
        ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2, color=plot_colour)
    # legend
    ax.legend(loc='lower right')
    return fig

def plot_tra(xdata, ydata, num_agents):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()

    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), \
        ax.transData)

    ax.plot(xdata, ydata, alpha=0.5)
    ax.axhline(y=0.5*int(num_agents), color='red', linestyle='dashed', alpha=0.5)
    ax.text(0.5, 0.4+(0.5*int(num_agents)), "0.5*N="+"{:.0f}".format(0.5*int(num_agents)), \
        color="red", transform=trans, ha="center", va="center", size=14)
    
    #ax.set_title('')
    ax.set_xlabel('Percentage of Target Performance (p)', fontsize=14)
    ax.set_ylabel('TTp(SingleLLAgent) / TTp(Shell)', fontsize=14)
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

def cfi_delta(data, conf_int_param=0.90): # confidence interval
    mean = np.mean(data, axis=0)
    if data.ndim == 1:
        std_error_of_mean = st.sem(data, axis=0)
        lb, ub = st.t.interval(conf_int_param, df=len(data)-1, loc=mean, scale=std_error_of_mean)
        cfi_delta = ub - mean
    elif data.ndim == 2:
        std_error_of_mean = st.sem(data, axis=0)
        print(std_error_of_mean)
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
        data['icr']['ll'] = {}
        data['icr']['ll']['xdata'] = np.arange(num_evals)
        data['icr']['ll']['ydata'] = np.mean(ll_icr, axis=0) # average across seeds
        data['icr']['ll']['ydata_cfi'] = np.std(ll_icr, axis=0)
        data['icr']['ll']['plot_colour'] = 'red'

        #print(data['icr']['ll']['ydata'])
        maximum_icr_ = np.max(data['icr']['ll']['ydata'])

        #print(data['icr']['ll']['xdata'])
        #print(data['icr']['ll']['ydata'])
        # tpot
        data['tpot']['ll'] = {}
        data['tpot']['ll']['xdata'] = np.arange(num_evals)
        data['tpot']['ll']['ydata'] = np.mean(ll_tpot, axis=0) # average across seeds
        data['tpot']['ll']['ydata_cfi'] = np.std(ll_tpot, axis=0)
        data['tpot']['ll']['plot_colour'] = 'red'

    # load shell data
    shell_data = []
    shell_icr = []
    shell_tpot = []
    for p in args.shell_paths:
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
    data['icr']['shell'] = {}
    data['icr']['shell']['xdata'] = np.arange(num_evals)
    data['icr']['shell']['ydata'] = np.mean(shell_icr, axis=0) # average across seeds
    cfi = cfi_delta(shell_icr)
   #print(cfi)
    data['icr']['shell']['ydata_cfi'] = cfi#np.std(shell_icr, axis=0)
    

    # Confidence Interval implementation. Overwrites the standard deviation data.
    #dof = len(data['icr']['shell']['xdata']) - 1
    #confidence = 0.95
    #t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    #data['icr']['shell']['ydata_cfi'] = (data['icr']['shell']['ydata']-data['icr']['shell']['ydata_cfi'] * t_crit / np.sqrt(len(data['icr']['shell']['xdata'])), data['icr']['shell']['ydata']-data['icr']['shell']['ydata_cfi'] * t_crit / np.sqrt(len(data['icr']['shell']['xdata'])))


    data['icr']['shell']['plot_colour'] = 'green'

    maximum_icr_ = max(maximum_icr_, np.max(data['icr']['shell']['ydata']))

    # tpot
    data['tpot']['shell'] = {}
    data['tpot']['shell']['xdata'] = np.arange(num_evals)
    data['tpot']['shell']['ydata'] = np.mean(shell_tpot, axis=0) # average across seeds
    data['tpot']['shell']['ydata_cfi'] = np.std(shell_tpot, axis=0)
    data['tpot']['shell']['plot_colour'] = 'green'


    
    ### BAR CHARTS
    '''
    ### Performance at 20%, 50% and 70% of MAX
    #print(data['icr']['shell']['ydata'])
    #print('max icr')
    #print(maximum_icr_)
    #print(np.where(data['icr']['shell']['ydata'] >= maximum_icr_)[0])
    #print('Here')
    _max_time_shell = data['icr']['shell']['xdata'][np.where(data['icr']['shell']['ydata'] >= maximum_icr_)[0][0]]
    _max_time_ll = data['icr']['ll']['xdata'][np.where(data['icr']['ll']['ydata'] >= maximum_icr_)[0][0]]
    _seventy = 0.70 * maximum_icr_
    _fifty = 0.50 * maximum_icr_
    _twenty= 0.20 * maximum_icr_
    bar_x = []
    bar_x.append(np.where(data['icr']['shell']['ydata'] >= _twenty)[0][0])
    bar_x.append(np.where(data['icr']['shell']['ydata'] >= _fifty)[0][0])
    bar_x.append(np.where(data['icr']['shell']['ydata'] >= _seventy)[0][0])

    bar_x2 = []
    bar_x2.append(np.where(data['icr']['ll']['ydata'] >= _twenty)[0][0])
    bar_x2.append(np.where(data['icr']['ll']['ydata'] >= _fifty)[0][0])
    bar_x2.append(np.where(data['icr']['ll']['ydata'] >= _seventy)[0][0])

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    xlabels = ['20%', '50%', '70%']
    ax.bar(xlabels, bar_x2, label='ll', width=0.6)
    ax.bar(xlabels, bar_x, label='shell', width=0.6)
    for i in range(len(xlabels)):
        ax.text(i, bar_x[i], str(bar_x[i]) + ' (' + str((round((bar_x[i]/_max_time_ll)*100, 2))) + '%)', ha = 'center', fontsize=15)
    for i in range(len(xlabels)):
        ax.text(i, bar_x2[i], str(bar_x2[i]) + ' (' + str((round((bar_x2[i]/_max_time_ll)*100, 2))) + '%)', ha = 'center', fontsize=15)
    
    plt.ylabel('Evaluation checkpoints', fontsize=15)
    plt.xlabel("X% of max performance", fontsize=15)
    plt.legend(loc='upper left')
    fig.savefig(save_path + 'bar_chart100.pdf', dpi=256, format='pdf', bbox_inches='tight')




    ### Performance at 20%, 50% and 70% of 95% performance
    maximum_icr_ninetyfive = 0.95 * maximum_icr_
    _max_time_shell = data['icr']['shell']['xdata'][np.where(data['icr']['shell']['ydata'] >= maximum_icr_ninetyfive)[0][0]]
    _max_time_ll = data['icr']['ll']['xdata'][np.where(data['icr']['ll']['ydata'] >= maximum_icr_ninetyfive)[0][0]]
    print(_max_time_shell, _max_time_ll)
    _seventy = 0.70 * _max_time_ll
    _fifty = 0.50 * _max_time_ll
    _twenty= 0.20 * _max_time_ll

    bar_x = []
    bar_x.append((data['icr']['shell']['ydata'][np.where(data['icr']['shell']['xdata'] >= _twenty)[0][0]] / maximum_icr_)*100)
    bar_x.append((data['icr']['shell']['ydata'][np.where(data['icr']['shell']['xdata'] >= _fifty)[0][0]] / maximum_icr_)*100)
    bar_x.append((data['icr']['shell']['ydata'][np.where(data['icr']['shell']['xdata'] >= _seventy)[0][0]] / maximum_icr_)*100)

    bar_x2 = []
    bar_x2.append((data['icr']['ll']['ydata'][np.where(data['icr']['ll']['xdata'] >= _twenty)[0][0]] / maximum_icr_)*100)
    bar_x2.append((data['icr']['ll']['ydata'][np.where(data['icr']['ll']['xdata'] >= _fifty)[0][0]] / maximum_icr_)*100)
    bar_x2.append((data['icr']['ll']['ydata'][np.where(data['icr']['ll']['xdata'] >= _seventy)[0][0]] / maximum_icr_)*100)
    print(_seventy, _fifty, _twenty)
    print(bar_x, bar_x2)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    xlabels = ['20%', '50%', '70%']
    ax.bar(xlabels, bar_x, label='shell', width=0.6)
    ax.bar(xlabels, bar_x2, label='ll', width=0.6)
    for i in range(len(xlabels)):
        ax.text(i, bar_x[i], str(round(bar_x[i], 2)) + '%', ha = 'center', fontsize=15)
    for i in range(len(xlabels)):
        ax.text(i, bar_x2[i], str(round(bar_x2[i], 2)) + '%', ha = 'center', fontsize=15)

    plt.ylabel('Performance (%)', fontsize=15)
    plt.xlabel("X% of time to reach 95% performance in the ll agent", fontsize=15)
    plt.legend(loc='upper left')
    fig.savefig(save_path + 'bar_chart95.pdf', dpi=256, format='pdf', bbox_inches='tight')
    '''



    # plot icr
    fig = plot(data['icr'], 'ICR', yaxis_label='Instant Cumulative Reward (ICR)')
    fig.savefig(save_path + 'metrics_icr.pdf', dpi=256, format='pdf', bbox_inches='tight')
    # plot tpot
    fig = plot(data['tpot'], 'TPOT', yaxis_label='Total Performance Over Time (TPOT)')
    fig.savefig(save_path + 'metrics_tpot.pdf', dpi=256, format='pdf', bbox_inches='tight')

    if args.ll_paths is not None:
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
        fig.savefig(save_path + 'TRA_' + str(NUM_AGENTS) + '_Agents.pdf', bbox_inches='tight', \
            dpi=256, format='pdf')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('shell_paths', help='paths to the experiment folder (support'\
        'paths to multiple seeds)', nargs='+')
    parser.add_argument('--ll_paths', help='paths to the experiment folder for single'\
        'agent lifelong learning (support paths to multiple seeds)', nargs='+', default=None)
    parser.add_argument('--exp_name', help='name of experiment', default='metrics_plot')
    parser.add_argument('--num_agents', help='number of agents in the experiment', type=int, default=1)
    parser.add_argument('--interval', help='interval', type=int, default=1)
    main(parser.parse_args())

