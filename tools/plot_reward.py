import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import glob
import argparse

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
    ax.set_ylim(0.0, 1.0)
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
        
    for method_name, result_dict in master.items():

        print(method_name)
        
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        plot_colour = result_dict['plot_colour']
        ax.plot(xdata, ydata, linewidth=3, label=method_name, alpha=0.5)
        ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2)
    # legend
    ax.legend(loc='lower right', prop={'size': 15})
    return fig

def plot_sum(fig, ax, master, title='', xaxis_label='Iteration', yaxis_label='Cumulative Return'):
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

mypaths = {
    ############## MCTGRAPH
    # fullcomm
    'mctgraph_full_comm' : {
        'Level 0' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T0/',
        'Level 1' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T1/',
        'Level 2' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T2/',
        'Level 3' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T3/',
        'Level 4' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T4/',
        'Level 5' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T5/',
        'Level 6' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T6/',
        'Level 7' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T7/',
        'Level 8' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T8/',
        'Level 9' : 'AAAI_EXPERIMENTS/mctgraph/full_comm/T9/'
    },

    # nocomm
    'mctgraph_no_comm' : {
        'Level 0' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T0/',
        'Level 1' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T1/',
        'Level 2' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T2/',
        'Level 3' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T3/',
        'Level 4' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T4/',
        'Level 5' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T5/',
        'Level 6' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T6/',
        'Level 7' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T7/',
        'Level 8' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T8/',
        'Level 9' : 'AAAI_EXPERIMENTS/mctgraph/no_comm/T9/'
    }
}
"""    ############## MINIGRID
    # fullcomm
    'minigrid_full_comm' : {
        'Level 0' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T0/',
        'Level 1' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T1/',
        'Level 2' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T2/',
        'Level 3' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T3/',
        'Level 4' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T4/',
        'Level 5' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T5/',
        'Level 6' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T6/',
        'Level 7' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T7/',
        'Level 8' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T8/',
        'Level 9' : 'AAAI_EXPERIMENTS/minigrid/full_comm/T9/'
    },

    # nocomm
    'minigrid_no_comm' : {
        'Level 0' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T0/',
        'Level 1' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T1/',
        'Level 2' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T2/',
        'Level 3' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T3/',
        'Level 4' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T4/',
        'Level 5' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T5/',
        'Level 6' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T6/',
        'Level 7' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T7/',
        'Level 8' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T8/',
        'Level 9' : 'AAAI_EXPERIMENTS/minigrid/no_comm/T9/'
    },

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
}"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_name', help='paths to the experiment folder for single'\
        'agent lifelong learning (support paths to multiple seeds)', type=str, default=None)
    parser.add_argument('--exp_name', help='name of experiment', default='metrics_plot')
    parser.add_argument('--num_agents', help='number of agents in the experiment', type=int, nargs='+', default=1)
    parser.add_argument('--interval', help='interval', type=int, default=1)
    args = parser.parse_args()


    fig2 = plt.figure(figsize=(30, 6))
    ax2 = fig2.subplots()

    # Set up axis labels, fonts, and limits
    ax2.set_xlabel('Iteration')
    ax2.xaxis.label.set_fontsize(20)
    ax2.set_ylabel('Cumulative Return')
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

    master = {}
    for plot_name, paths in mypaths.items():
        print('NAMES:', plot_name, 'PATHS:', paths)
        for name, path in paths.items():
            data = pd.DataFrame()
            #if path[-1] != '/': path += '/'
            for i, filepath in enumerate(sorted(glob.glob(f'{path}*.csv'))):
                print(i, filepath)
                # Load data into a pandas dataframe
                df = pd.read_csv(filepath)
                # Select data from second column for each seed run
                data.loc[:, i] = df['Value']

            print('DATA', name)
            print(data)

            master[name] = {}
            master[name]['xdata'] = np.arange(data.shape[0])
            master[name]['ydata'] = np.mean(data, axis=1)
            print('MASTER', name)
            print(master[name]['ydata'])
            master[name]['ydata_cfi'] = cfi_delta(data)
            master[name]['plot_colour'] = 'green'


        if not os.path.exists('./log/plots/'):
                os.makedirs('./log/plots/')
        fig1 = plot(master, yaxis_label='Return')
        fig1.savefig(f'./log/plots/{plot_name}.pdf', dpi=256, format='pdf', bbox_inches='tight')

        fig2, ax2 = plot_sum(fig2, ax2, master, title=plot_name, yaxis_label='Instant Cumulative Return')
    fig2.savefig(f'./log/plots/cumulative.pdf', dpi=256, format='pdf', bbox_inches='tight')