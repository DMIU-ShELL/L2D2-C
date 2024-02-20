import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os

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

def plot(master, title='', xaxis_label='Iterations', yaxis_label=''):
    #fig = plt.figure(figsize=(25, 6))  # For wide graph
    fig = plt.figure(figsize=(25, 6))
    
    ax = fig.subplots()



    ax.set_xlabel(xaxis_label)
    ax.xaxis.label.set_fontsize(20) # Originally 30
    ax.set_ylabel(yaxis_label)
    ax.yaxis.label.set_fontsize(20) # Originally 30
    ax.set_ylim(0, 1.1)
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

# Depth 2
"""mypaths = {
        'Full Communication' : ['Science Robotics Data/Full Comm/Depth 2/full_comm_agent_0_MetaCTgraph-shell-dist-ct8_md_agent1-seed-9157_agent_0_231213-154936.csv',
                                'Science Robotics Data/Full Comm/Depth 2/full_comm_agent_0_MetaCTgraph-shell-dist-ct8_md_agent1-seed-9802_agent_0_231213-155737.csv',
                                'Science Robotics Data/Full Comm/Depth 2/full_comm_agent_0_MetaCTgraph-shell-dist-ct8_md_agent1-seed-9822_agent_0_231213-160449.csv'],

        'No Communication' : ['Science Robotics Data/No Comm/Depth 2/no_comm_agent_0_MetaCTgraph-shell-dist-ct8_md_agent1-seed-9157_agent_0_231213-163721.csv',
                                'Science Robotics Data/No Comm/Depth 2/no_comm_agent_0_MetaCTgraph-shell-dist-ct8_md_agent1-seed-9802_agent_0_231213-162158.csv',
                                'Science Robotics Data/No Comm/Depth 2/no_comm_agent_0_MetaCTgraph-shell-dist-ct8_md_agent1-seed-9822_agent_0_231213-161529.csv'],
    }

# Depth 3
mypaths = {
        'Full Communication' : ['Science Robotics Data/Full Comm/Depth 3/full_comm_agent_1_MetaCTgraph-shell-dist-ct8_md_agent2-seed-6302_agent_1_231213-155743.csv',
                                'Science Robotics Data/Full Comm/Depth 3/full_comm_agent_1_MetaCTgraph-shell-dist-ct8_md_agent2-seed-8946_agent_1_231213-160449.csv',
                                'Science Robotics Data/Full Comm/Depth 3/full_comm_agent_1_MetaCTgraph-shell-dist-ct8_md_agent2-seed-9158_agent_1_231213-154936.csv'],

        'No Communication' : ['Science Robotics Data/No Comm/Depth 3/no_comm_agent_1_MetaCTgraph-shell-dist-ct8_md_agent2-seed-6302_agent_1_231213-162159.csv',
                                'Science Robotics Data/No Comm/Depth 3/no_comm_agent_1_MetaCTgraph-shell-dist-ct8_md_agent2-seed-8946_agent_1_231213-161529.csv',
                                'Science Robotics Data/No Comm/Depth 3/no_comm_agent_1_MetaCTgraph-shell-dist-ct8_md_agent2-seed-9158_agent_1_231213-163721.csv'],
    }

# Depth 4
mypaths = {
        'Full Communication' : ['Science Robotics Data/Full Comm/Depth 4/full_comm_agent_2_MetaCTgraph-shell-dist-ct8_md_agent3-seed-1902_agent_2_231213-155743.csv',
                                'Science Robotics Data/Full Comm/Depth 4/full_comm_agent_2_MetaCTgraph-shell-dist-ct8_md_agent3-seed-4693_agent_2_231213-160449.csv',
                                'Science Robotics Data/Full Comm/Depth 4/full_comm_agent_2_MetaCTgraph-shell-dist-ct8_md_agent3-seed-9159_agent_2_231213-154936.csv'],

        'No Communication' : ['Science Robotics Data/No Comm/Depth 4/no_comm_agent_2_MetaCTgraph-shell-dist-ct8_md_agent3-seed-1902_agent_2_231213-162158.csv',
                                'Science Robotics Data/No Comm/Depth 4/no_comm_agent_2_MetaCTgraph-shell-dist-ct8_md_agent3-seed-4693_agent_2_231213-161529.csv',
                                'Science Robotics Data/No Comm/Depth 4/no_comm_agent_2_MetaCTgraph-shell-dist-ct8_md_agent3-seed-9159_agent_2_231213-163722.csv'],
    }

# Depth 5
mypaths = {
        'Full Communication' : ['Science Robotics Data/Full Comm/Depth 5/full_comm_agent_3_MetaCTgraph-shell-dist-ct8_md_agent4-seed-3519_agent_3_231213-160450.csv',
                                'Science Robotics Data/Full Comm/Depth 5/full_comm_agent_3_MetaCTgraph-shell-dist-ct8_md_agent4-seed-4446_agent_3_231213-155744.csv',
                                'Science Robotics Data/Full Comm/Depth 5/full_comm_agent_3_MetaCTgraph-shell-dist-ct8_md_agent4-seed-9160_agent_3_231213-154936.csv'],

        'No Communication' : ['Science Robotics Data/No Comm/Depth 5/no_comm_agent_3_MetaCTgraph-shell-dist-ct8_md_agent4-seed-3519_agent_3_231213-161529.csv',
                                'Science Robotics Data/No Comm/Depth 5/no_comm_agent_3_MetaCTgraph-shell-dist-ct8_md_agent4-seed-4446_agent_3_231213-162158.csv',
                                'Science Robotics Data/No Comm/Depth 5/no_comm_agent_3_MetaCTgraph-shell-dist-ct8_md_agent4-seed-9160_agent_3_231213-163721.csv'],
    }"""


"""mypaths = {
        'depth 2' : ['Initial_Results_Data/seed1_MetaCTgraph-shell-dist-ct28_agent1_left-seed-9157_agent_0_240209-150346.csv',
                     'Initial_Results_Data/seed2_MetaCTgraph-shell-dist-ct28_agent1_left-seed-9802_agent_0_240210-124604.csv'],
        'depth 3' : ['Initial_Results_Data/seed1_MetaCTgraph-shell-dist-ct28_agent2_left-seed-9158_agent_1_240209-150346.csv',
                     'Initial_Results_Data/seed2_MetaCTgraph-shell-dist-ct28_agent2_left-seed-6302_agent_1_240210-124604.csv'],
        'depth 4' : ['Initial_Results_Data/seed1_MetaCTgraph-shell-dist-ct28_agent3_left-seed-9159_agent_2_240209-150346.csv',
                     'Initial_Results_Data/seed2_MetaCTgraph-shell-dist-ct28_agent3_left-seed-1902_agent_2_240210-124604.csv'],
        'depth 5' : ['Initial_Results_Data/seed1_MetaCTgraph-shell-dist-ct28_agent4_left-seed-9160_agent_3_240209-150346.csv',
                     'Initial_Results_Data/seed2_MetaCTgraph-shell-dist-ct28_agent4_left-seed-4446_agent_3_240210-124604.csv'],
        'depth 6' : ['Initial_Results_Data/seed1_MetaCTgraph-shell-dist-ct28_agent5_left-seed-9161_agent_4_240209-150346.csv',
                     'Initial_Results_Data/seed2_MetaCTgraph-shell-dist-ct28_agent5_left-seed-9575_agent_4_240210-124604.csv'],
        'depth 7' : ['Initial_Results_Data/seed1_MetaCTgraph-shell-dist-ct28_agent6_left-seed-9162_agent_5_240209-150346.csv',
                     'Initial_Results_Data/seed2_MetaCTgraph-shell-dist-ct28_agent6_left-seed-1954_agent_5_240210-124604.csv'],
    }"""


mypaths = {
        'depth 2' : ['Initial_Results_Data/no_comm/seed1_MetaCTgraph-shell-dist-ct28_agent1_left-seed-9157_agent_0_240209-164032.csv',
                     'Initial_Results_Data/no_comm/seed2_MetaCTgraph-shell-dist-ct28_agent1_left-seed-9802_agent_0_240210-155928.csv'],
        'depth 3' : ['Initial_Results_Data/no_comm/seed1_MetaCTgraph-shell-dist-ct28_agent2_left-seed-9158_agent_1_240209-164033.csv',
                     'Initial_Results_Data/no_comm/seed2_MetaCTgraph-shell-dist-ct28_agent2_left-seed-6302_agent_1_240210-155928.csv'],
        'depth 4' : ['Initial_Results_Data/no_comm/seed1_MetaCTgraph-shell-dist-ct28_agent3_left-seed-9159_agent_2_240209-164032.csv',
                     'Initial_Results_Data/no_comm/seed2_MetaCTgraph-shell-dist-ct28_agent3_left-seed-1902_agent_2_240210-155928.csv'],
        'depth 5' : ['Initial_Results_Data/no_comm/seed1_MetaCTgraph-shell-dist-ct28_agent4_left-seed-9160_agent_3_240209-164032.csv',
                     'Initial_Results_Data/no_comm/seed2_MetaCTgraph-shell-dist-ct28_agent4_left-seed-4446_agent_3_240210-155928.csv'],
        'depth 6' : ['Initial_Results_Data/no_comm/seed1_MetaCTgraph-shell-dist-ct28_agent5_left-seed-9161_agent_4_240209-164032.csv',
                     'Initial_Results_Data/no_comm/seed2_MetaCTgraph-shell-dist-ct28_agent5_left-seed-9575_agent_4_240210-155928.csv'],
        'depth 7' : ['Initial_Results_Data/no_comm/seed1_MetaCTgraph-shell-dist-ct28_agent6_left-seed-9162_agent_5_240209-164032.csv',
                     'Initial_Results_Data/no_comm/seed2_MetaCTgraph-shell-dist-ct28_agent6_left-seed-1954_agent_5_240210-155928.csv'],
    }


master = {}
for name, paths in mypaths.items():
    data = pd.DataFrame()
    for i, path in enumerate(paths):
        
        #print(name, path, i)

        # Load data into a pandas dataframe
        df = pd.read_csv(path)

        #print(df)
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
fig = plot(master, yaxis_label='Return')
fig.savefig('./log/plots/agent3_depth5.pdf', dpi=256, format='pdf', bbox_inches='tight')