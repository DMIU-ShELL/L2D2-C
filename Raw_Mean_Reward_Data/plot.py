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
    fig = plt.figure(figsize=(25, 6))
    ax = fig.subplots()



    ax.set_xlabel(xaxis_label)
    ax.xaxis.label.set_fontsize(30)
    ax.set_ylabel(yaxis_label)
    ax.yaxis.label.set_fontsize(30)
    ax.set_ylim(0, 1.2)
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
    ax.legend(loc='lower right')
    return fig

mypaths = {
        'DDPPO'  : ['Raw_Mean_Reward_Data/DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_0adbd_00000_0_2023-02-27_15-04-33_WORKERS_2_SEED_959.csv', 
                    'Raw_Mean_Reward_Data/DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_16dcf_00000_0_2023-02-27_17-06-34_WORKERS_2_SEED_960.csv', 
                    'Raw_Mean_Reward_Data/DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_16e7d_00000_0_2023-02-27_14-57-44_WORKERS_2_SEED_958.csv', 
                    'Raw_Mean_Reward_Data/DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_87b2f_00000_0_2023-02-27_23-00-29_WORKERS_2_SEED_962.csv', 
                    'Raw_Mean_Reward_Data/DDPPO/2_Workers/Trainees/DDPPO_CT-Graph_d2_b2_wp05_crv0_MDP_8478f_00000_0_2023-02-27_17-09-38_WORKERS_2_SEED_961.csv'],
                    
        'IMPALA' : ['Raw_Mean_Reward_Data/IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_6f121_00000_0_2023-03-01_18-16-34_WORKERS_2_SEED_959.csv', 
                    'Raw_Mean_Reward_Data/IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_9070f_00000_0_2023-03-01_18-17-30_WORKERS_2_SEED_960.csv', 
                    'Raw_Mean_Reward_Data/IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_9891f_00000_0_2023-03-01_17-27-37_WORKERS_2_SEED_958.csv', 
                    'Raw_Mean_Reward_Data/IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_b7543_00000_0_2023-03-01_18-18-35_WORKERS_2_SEED_961.csv', 
                    'Raw_Mean_Reward_Data/IMPALA/2_Workers/Trainees/IMPALA_CT-Graph_d2_b2_wp05_crv0_MDP_e7696_00000_0_2023-03-01_18-19-55_WORKERS_2_SEED_962.csv'],

        'PPO'    : ['Raw_Mean_Reward_Data/PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_8b766_00000_0_2023-02-27_23-07-45_SEED_962.csv', 
                    'Raw_Mean_Reward_Data/PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_00221_00000_0_2023-03-02_13-49-06_SEED_958.csv', 
                    'Raw_Mean_Reward_Data/PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_cf138_00000_0_2023-02-27_15-10-02_SEED_960.csv',
                    'Raw_Mean_Reward_Data/PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_d6da2_00000_0_2023-02-27_15-03-06_SEED_959.csv', 
                    'Raw_Mean_Reward_Data/PPO/1_Workers/Trainees/PPO_CT-Graph_d2_b2_wp05_crv0_MDP_ecece_00000_0_2023-02-27_22-56-09_SEED_961.csv'],

        'L2D2-C' : ['Raw_Mean_Reward_Data/ShELL/2_Agents/Agent1/run-MetaCTgraph-shell-dist-upz-seed-962_agent_1_230310-002012-tag-agent_1_iteration_avg_reward.csv', 
                    'Raw_Mean_Reward_Data/ShELL/2_Agents/Agent2/run-MetaCTgraph-shell-dist-upz-seed-959_agent_2_230308-131206-tag-agent_2_iteration_avg_reward.csv']
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



if not os.path.exists('./log/plots/baselines/'):
        os.makedirs('./log/plots/baselines/')
fig = plot(master, yaxis_label='Return')
fig.savefig('./log/plots/baselines/baselines.pdf', dpi=256, format='pdf', bbox_inches='tight')