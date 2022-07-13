#-*- coding: utf-8 -*-
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot(results, title='', yaxis_label=''):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    # axis title and font
    ax.set_title(title)
    ax.title.set_fontsize(22)
    # axis labels and font, and ticks
    ax.set_xlabel('Number of evaluations ')
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
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        plot_colour = result_dict['plot_colour']
        ax.plot(xdata, ydata, linewidth=3, label=method_name, color=plot_colour)
        ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2, color=plot_colour)
    # legend
    ax.legend(loc='lower right')
    return fig

def load_shell_data(args_path):
    if args_path[-1] != '/':
        args_path += '/'

    names = os.listdir(args_path)
    names = [name for name in names if re.search('agent_*', name) is not None]
    names = sorted(names, key=lambda x: int(x.split('_')[1]))

    paths = [args_path + name for name in names]
    paths = ['{0}/{1}/'.format(path, os.listdir(path)[-1]) for path in paths]
    num_agents = len(paths)
    metrics = []
    for name, path in zip(names, paths):
        file_path = path + 'eval_metrics_{0}.csv'.format(name)
        m = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
        if m.ndim == 1:
            m = np.expand_dims(m, axis=0)
        metrics.append(m)
        #print(m.shape)
    num_evals = metrics[0].shape[0]
    num_tasks = metrics[0].shape[1]
    
    # shape: num_agents x num_evals x num_tasks
    metrics = np.stack(metrics, axis=0)
    #print(metrics.shape)
    # shape: num_evals x num_agents x num_tasks
    metrics = metrics.transpose(1, 0, 2)
    #print(metrics.shape)
    
    metrics_icr = []
    metrics_tpot = []
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
    
    return metrics, metrics_icr, metrics_tpot

def load_ll_data(path):
        # Load baseline data for a single LL agent
        if os.path.exists(path + 'eval_metrics.csv'):
            path = path + 'eval_metrics.csv'
            raw_data = np.loadtxt(path, dtype=np.float32, delimiter=',')
        else:
            path = path + 'eval/eval_metrics.npy'
            raw_data = np.load(path) # shape: num_evals x num_tasks
        metrics_icr = raw_data.sum(axis=1)
        metrics_tpot = []
        for idx in range(len(metrics_icr)):
            metrics_tpot.append(sum(metrics_icr[0 : idx]))
        metrics_tpot = np.asarray(metrics_tpot)
        return raw_data, metrics_icr, metrics_tpot

def main(args):
    exp_name = 'MiniGrid-shell-dist'
    #save_path = './log/plots/' + os.path.basename(args.path[ : -1]) + '/'
    save_path = './log/plots/' + exp_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = {}
    data['icr'] = {}
    data['tpot'] = {}
    data['sbf1'] = {}
    data['sbf2'] = {}
    data['sbf3'] = {}

    # load single agent (ll) data if it exists
    if args.ll_paths is not None:
        ll_data = []
        ll_icr = []
        ll_tpot = []
        for p in args.ll_paths:
            raw_data, metrics_icr, metrics_tpot = load_ll_data(p)
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
        raw_data, metrics_icr, metrics_tpot = load_shell_data(p)
        shell_data.append(raw_data)
        shell_icr.append(metrics_icr)
        shell_tpot.append(metrics_tpot)
    shell_data = np.stack(shell_data, axis=0) # shape: num_seeds x num_evals x num_agents x num_tasks
    shell_icr = np.stack(shell_icr, axis=0)  # shape: num_seeds x num_evals
    shell_tpot = np.stack(shell_tpot, axis=0) # shape: num_seeds x num_evals
    num_evals = shell_data.shape[1]
	# icr
    data['icr']['shell'] = {}
    data['icr']['shell']['xdata'] = np.arange(num_evals)
    data['icr']['shell']['ydata'] = np.mean(shell_icr, axis=0) # average across seeds
    data['icr']['shell']['ydata_cfi'] = np.std(shell_tpot, axis=0)
    data['icr']['shell']['plot_colour'] = 'green'
    # tpot
    data['tpot']['shell'] = {}
    data['tpot']['shell']['xdata'] = np.arange(num_evals)
    data['tpot']['shell']['ydata'] = np.mean(shell_tpot, axis=0) # average across seeds
    data['tpot']['shell']['ydata_cfi'] = np.std(shell_tpot, axis=0)
    data['tpot']['shell']['plot_colour'] = 'green'

    # plot icr
    fig = plot(data['icr'], 'Instant Cumulative Reward (ICR)', 'ICR')
    fig.savefig(save_path + 'metrics_icr.pdf', dpi=256, format='pdf')
    # plot tpot
    fig = plot(data['tpot'], 'Total Performance Over Time (TPOT)', 'TPOT')
    fig.savefig(save_path + 'metrics_tpot.pdf', dpi=256, format='pdf')

    if args.ll_paths is not None:
        eps = 1e-6 # to help with zero divide
        # sbf 1
        sbf = data['tpot']['shell']['ydata'] / (data['tpot']['ll']['ydata'] + eps)
        data['sbf1']['shell'] = {}
        data['sbf1']['shell']['xdata'] = np.arange(num_evals)
        data['sbf1']['shell']['ydata'] = sbf
        data['sbf1']['shell']['ydata_cfi'] = np.zeros_like(sbf)
        data['sbf1']['shell']['plot_colour'] = 'green'
        # sbf 2
        sbf = data['icr']['shell']['ydata'] / (data['icr']['ll']['ydata'] + eps)
        data['sbf2']['shell'] = {}
        data['sbf2']['shell']['xdata'] = np.arange(num_evals)
        data['sbf2']['shell']['ydata'] = sbf 
        data['sbf2']['shell']['ydata_cfi'] = np.zeros_like(sbf)
        data['sbf2']['shell']['plot_colour'] = 'green'
        # plot sbf1
        y_label = 'TPOT(Shell, t) / TPOT(SingleLLAgent, t)'
        fig = plot(data['sbf1'], 'Total Learning Advantage (TLA): SBF1', y_label)
        fig.savefig(save_path + 'metrics_tla.pdf', dpi=256, format='pdf')
        # plot sbf2
        y_label = 'ICR(Shell, t) / ICR(SingleLLAgent, t)'
        fig = plot(data['sbf2'], 'Instant Learning Advantage (ILA): SBF2', y_label)
        fig.savefig(save_path + 'metrics_ila.pdf', dpi=256, format='pdf')
        
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('shell_paths', help='paths to the experiment folder (support'\
        'paths to multiple seeds)', nargs='+')
    parser.add_argument('--ll_paths', help='paths to the experiment folder for single'\
        'agent lifelong learning', nargs='+', default=None)
    main(parser.parse_args())

