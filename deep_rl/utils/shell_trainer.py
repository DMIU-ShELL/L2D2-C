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


def shell_train(agents, logger):
    num_agents = len(agents)
    shell_done = [False,] * num_agents
    shell_iterations = [0,] * num_agents
    shell_tasks = [agent.config.cl_tasks_info for agent in agents] # tasks for each agent
    shell_task_idx = [0,] * num_agents

    print()
    logger.info('*****start shell training')

    # set the first task each agent is meant to train on
    for agent_idx, agent in enumerate(agents):
        states_ = agent.task.reset_task(shell_tasks[agent_idx][0])
        agent.states = agent.config.state_normalizer(states_)
        logger.info('*****setting first task for agent {0}'.format(agent_idx))
        logger.info('task: {0}'.format(shell_tasks[agent_idx][0]['task']))
        logger.info('task_label: {0}'.format(shell_tasks[agent_idx][0]['task_label']))
        agent.task_train_start(task_idx=0)
        print()
    del states_

    while True:
        for agent_idx, agent in enumerate(agents):
            if shell_done[agent_idx]:
                continue
            agent.iteration()
            shell_iterations[agent_idx] += 1
            if shell_iterations[agent_idx] % agent.config.iteration_log_interval == 0:
                logger.info('agent %d, task %d / iteration %d, total steps %d, ' \
                'mean/max/min reward %f/%f/%f' % (agent_idx, shell_task_idx[agent_idx], \
                    shell_iterations[agent_idx],
                    agent.total_steps,
                    np.mean(agent.last_episode_rewards),
                    np.max(agent.last_episode_rewards),
                    np.min(agent.last_episode_rewards)
                ))
                logger.scalar_summary('agent_{0}/avg_reward'.format(agent_idx), \
                    np.mean(agent.last_episode_rewards))
                logger.scalar_summary('agent_{0}/max_reward'.format(agent_idx), \
                    np.max(agent.last_episode_rewards))
                logger.scalar_summary('agent_{0}/min_reward'.format(agent_idx), \
                    np.min(agent.last_episode_rewards))

            task_steps_limit = agent.config.max_steps * (shell_task_idx[agent_idx] + 1)
            if agent.config.max_steps and agent.total_steps >= task_steps_limit:
                print()
                logger.info('*****agent {0} / end of training on task {1}'.format(agent_idx, \
                    shell_task_idx[agent_idx]))
                agent.task_train_end(shell_task_idx[agent_idx])

                shell_task_idx[agent_idx] += 1
                if shell_task_idx[agent_idx] < len(shell_tasks[agent_idx]):
                    task_idx_ = shell_task_idx[agent_idx]
                    logger.info('*****agent {0} / set next task {1}'.format(agent_idx, task_idx_))
                    logger.info('task: {0}'.format(shell_tasks[agent_idx][task_idx_]['task']))
                    logger.info('task_label: {0}'.format(shell_tasks[agent_idx][task_idx_]['task_label']))
                    states_ = agent.task.reset_task(shell_tasks[agent_idx][task_idx_]) # set new task
                    agent.states = agent.config.state_normalizer(states_)
                    agent.task_train_start(task_idx=task_idx_)

                    # ping other agents to see if they have knoweledge (mask) of current task
                    logger.info('*****agent {0} / pinging other agents to leverage on existing ' \
                        'knowledge about task')
                    a_idxs = list(range(len(agents)))
                    a_idxs.remove(agent_idx)
                    other_agents = [agents[i] for i in a_idxs]
                    agent.ping_agents(other_agents)
                    del states_
                    del task_idx_
                    print()
                else:
                    shell_done[agent_idx] = True # training done for all task for agent
                    
        if all(shell_done):
            break
    for agent in agents:
        agent.close()
    return
