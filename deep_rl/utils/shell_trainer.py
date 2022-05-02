#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import datetime
import torch
from .torch_utils import *
#from io import BytesIO
#import scipy.misc
#import torchvision

try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path


def shell_train(agents):
    num_agents = len(agents)
    shell_done = [False,] * num_agents
    shell_iterations = [0,] * num_agents
    shell_tasks = [agent.config.cl_tasks_info for agent in agents] # tasks for each agent
    shell_task_idx = [0,] * num_agents

    config.logger.info('**********start shell training')

    # set the first task each agent is meant to train on
    for agent_idx, agent in enumerate(agents):
        states_ = agent.reset_task(shell_tasks[agent_idx][0])
        agent.states = config.state_normalizer(states_)
    del states_

    while True:
        for agent_idx, agent in enumerate(agents):
            if shell_done[agent_idx]:
                break
            agent.iteration()
            shell_iterations[agent_idx] += 1
            if shell_iterations[agent_idx] % config.iteration_log_interval == 0:
                config.logger.info('agent %d, iteration %d, total steps %d, mean/max/min reward'+\
                    '%f/%f/%f' % (agent_idx, shell_iteration[agent_idx], agent.total_steps, \
                    np.mean(agent.last_episode_rewards),
                    np.max(agent.last_episode_rewards),
                    np.min(agent.last_episode_rewards)
                ))
                config.logger.scalar_summary('agent_{0}/avg_reward'.format(agent_idx, \
                    np.mean(agent.last_episode_rewards)))
                config.logger.scalar_summary('agent_{0}/max_reward'.format(agent_idx, \
                    np.max(agent.last_episode_rewards)))
                config.logger.scalar_summary('agent_{0}/min_reward'.format(agent_idx, \
                    np.min(agent.last_episode_rewards)))

            task_steps_limit = config.max_steps * (task_idx + 1)
            if config.max_steps and agent.total_steps >= task_steps_limit:
                shell_task_idx[agent_idx] += 1
                if shell_task_idx[agent_idx] < len(shell_tasks[agent_idx]):
                    config.logger.info('**********set next task {0}'.format(task_idx))
                    config.logger.info('task: {0}'.format(shell_task[agent_idx]['task']))
                    config.logger.info('task_label: {0}'.format(shell_task[agent_idx]['task_label']))
                    task_idx_ = shell_task_idx[agent_idx]
                    states_ = agent.reset_task(shell_tasks[agent_idx][task_idx_]) # set new task
                    agent.states = config.state_normalizer(states_)
                    del states_
                    del task_idx_
                else:
                    shell_done[agent_idx] = True # training done for all task for agent
                    
        if all(shell_done):
            break
    for agent in agents:
        agent.close()
    return
