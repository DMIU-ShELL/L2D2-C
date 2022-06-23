#-*- coding: utf-8 -*-
import os
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
        plot_colour = result_dict['plot_colour']
        ax.plot(xdata, ydata, linewidth=3, label=method_name, color=plot_colour)
        #ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2, color=plot_colour)
    return fig

def main(args):
    if args.path[-1] != '/':
        args.path += '/'
    save_path = './plots/' + os.path.basename(args.path[ : -1]) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    names = os.listdir(args.path)
    names = sorted(names, key=lambda x: int(x.split('_')[1]))

    paths = [args.path + name for name in names]
    paths = ['{0}/{1}/'.format(path, os.listdir(path)[-1]) for path in paths]
    num_agents = len(paths)
    metrics = []
    for name, path in zip(names, paths):
        file_path = path + 'eval_metrics_{0}.csv'.format(name)
        m = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
        if m.ndim == 1:
            m = np.expand_dims(m, axis=0)
        metrics.append(m)
        print(m.shape)
    num_evals = metrics[0].shape[0]
    num_tasks = metrics[0].shape[1]

    # shape: num_agents x num_evals x num_tasks
    metrics = np.stack(metrics, axis=0)
    print(metrics.shape)
    # shape: num_evals x num_agents x num_tasks
    metrics = metrics.transpose(1, 0, 2)
    print(metrics.shape)

    metrics_icr = []
    metrics_tp = []
    for idx in range(num_evals):
        data = metrics[idx]
        _max_reward = data.max(axis=0)
        agent_ids = data.argmax(axis=0).tolist()
        print('best agent per task: {0}'.format(agent_ids))
        # compute icr/tcr
        icr = _max_reward.sum()
        metrics_icr.append(icr)

        tp = np.sum(metrics_icr)
        metrics_tp.append(tp)

    data = {}
    data['shell'] = {}
    data['shell']['xdata'] = np.arange(len(metrics_icr))
    data['shell']['ydata'] = np.asarray(metrics_icr)
    data['shell']['plot_colour'] = 'red'
    fig = plot(data, 'ICR')
    fig.savefig(save_path + 'metrics_icr.pdf', dpi=256, format='pdf')

    data = {}
    data['shell'] = {}
    data['shell']['xdata'] = np.arange(len(metrics_tp))
    data['shell']['ydata'] = np.asarray(metrics_tp)
    data['shell']['plot_colour'] = 'red'
    fig = plot(data, 'TP')
    fig.savefig(save_path + 'metrics_tp.pdf', dpi=256, format='pdf')
        
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to the experiment folder')
    main(parser.parse_args())

