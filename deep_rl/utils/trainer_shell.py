#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################



#   __________ .__                              __   .__     .__                                           
#   \______   \|  |    ____    ______  ______ _/  |_ |  |__  |__|  ______   _____    ____    ______  ______
#    |    |  _/|  |  _/ __ \  /  ___/ /  ___/ \   __\|  |  \ |  | /  ___/  /     \ _/ __ \  /  ___/ /  ___/
#    |    |   \|  |__\  ___/  \___ \  \___ \   |  |  |   Y  \|  | \___ \  |  Y Y  \\  ___/  \___ \  \___ \ 
#    |______  /|____/ \___  >/____  >/____  >  |__|  |___|  /|__|/____  > |__|_|  / \___  >/____  >/____  >
#           \/            \/      \/      \/              \/          \/        \/      \/      \/      \/ 
#
#                                                     :')


import numpy as np
import time
import torch
from .torch_utils import *
from tensorboardX import SummaryWriter
from ..shell_modules import *

import multiprocessing.dummy as mpd
from colorama import Fore
import psutil
import pandas as pd

try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path

from memory_profiler import profile


def _shell_itr_log(logger, agent, agent_idx, itr_counter, task_counter, dict_logs):
    logger.info(Fore.BLUE + 'agent %d, task %d / iteration %d, total steps %d, ' \
    'mean/max/min reward %f/%f/%f' % (agent_idx, task_counter, \
        itr_counter,
        agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards)
    ))
    logger.scalar_summary('agent_{0}/last_episode_avg_reward'.format(agent_idx), \
        np.mean(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_std_reward'.format(agent_idx), \
        np.std(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_max_reward'.format(agent_idx), \
        np.max(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_min_reward'.format(agent_idx), \
        np.min(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/iteration_avg_reward'.format(agent_idx), \
        np.mean(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_std_reward'.format(agent_idx), \
        np.std(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_max_reward'.format(agent_idx), \
        np.max(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_min_reward'.format(agent_idx), \
        np.min(agent.iteration_rewards))

    prefix = 'agent_{0}_'.format(agent_idx)
    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('{0}debug/{1}_avg_norm'.format(prefix, tag), np.mean(value_norm))
            logger.scalar_summary('{0}debug/{1}_avg'.format(prefix, tag), value.mean())
            logger.scalar_summary('{0}debug/{1}_std'.format(prefix, tag), value.std())
            logger.scalar_summary('{0}debug/{1}_max'.format(prefix, tag), value.max())
            logger.scalar_summary('{0}debug/{1}_min'.format(prefix, tag), value.min())


    #print(dict_logs)
    for key, value in dict_logs.items():
        #print(key, value)
        logger.scalar_summary('{0}debug_extended/{1}_avg'.format(prefix, key), np.mean(value))
        logger.scalar_summary('{0}debug_extended/{1}_std'.format(prefix, key), np.std(value))
        logger.scalar_summary('{0}debug_extended/{1}_max'.format(prefix, key), np.max(value))
        logger.scalar_summary('{0}debug_extended/{1}_min'.format(prefix, key), np.min(value))

    return

# metaworld/continualworld
def _shell_itr_log_mw(logger, agent, agent_idx, itr_counter, task_counter, dict_logs):
    logger.info(Fore.BLUE + 'agent %d, task %d / iteration %d, total steps %d, ' \
    'mean/max/min reward %f/%f/%f, mean/max/min success rate %f/%f/%f' % (agent_idx, \
        task_counter,
        itr_counter,
        agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards),
        np.mean(agent.iteration_success_rate),
        np.max(agent.iteration_success_rate),
        np.min(agent.iteration_success_rate)
    ))
    logger.scalar_summary('agent_{0}/last_episode_avg_reward'.format(agent_idx), \
        np.mean(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_std_reward'.format(agent_idx), \
        np.std(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_max_reward'.format(agent_idx), \
        np.max(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_min_reward'.format(agent_idx), \
        np.min(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/iteration_avg_reward'.format(agent_idx), \
        np.mean(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_std_reward'.format(agent_idx), \
        np.std(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_max_reward'.format(agent_idx), \
        np.max(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_min_reward'.format(agent_idx), \
        np.min(agent.iteration_rewards))

    logger.scalar_summary('agent_{0}/last_episode_avg_success_rate'.format(agent_idx), \
        np.mean(agent.last_episode_success_rate))
    logger.scalar_summary('agent_{0}/last_episode_std_success_rate'.format(agent_idx), \
        np.std(agent.last_episode_success_rate))
    logger.scalar_summary('agent_{0}/last_episode_max_success_rate'.format(agent_idx), \
        np.max(agent.last_episode_success_rate))
    logger.scalar_summary('agent_{0}/last_episode_min_success_rate'.format(agent_idx), \
        np.min(agent.last_episode_success_rate))
    logger.scalar_summary('agent_{0}/iteration_avg_success_rate'.format(agent_idx), \
        np.mean(agent.iteration_success_rate))
    logger.scalar_summary('agent_{0}/iteration_std_success_rate'.format(agent_idx), \
        np.std(agent.iteration_success_rate))
    logger.scalar_summary('agent_{0}/iteration_max_success_rate'.format(agent_idx), \
        np.max(agent.iteration_success_rate))
    logger.scalar_summary('agent_{0}/iteration_min_success_rate'.format(agent_idx), \
        np.min(agent.iteration_success_rate))

    prefix = 'agent_{0}_'.format(agent_idx)
    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('{0}debug/{1}_avg_norm'.format(prefix, tag), np.mean(value_norm))
            logger.scalar_summary('{0}debug/{1}_avg'.format(prefix, tag), value.mean())
            logger.scalar_summary('{0}debug/{1}_std'.format(prefix, tag), value.std())
            logger.scalar_summary('{0}debug/{1}_max'.format(prefix, tag), value.max())
            logger.scalar_summary('{0}debug/{1}_min'.format(prefix, tag), value.min())

    for key, value in dict_logs.items():
        logger.scalar_summary('{0}debug_extended/{1}_avg'.format(prefix, key), np.mean(value))
        logger.scalar_summary('{0}debug_extended/{1}_std'.format(prefix, key), np.std(value))
        logger.scalar_summary('{0}debug_extended/{1}_max'.format(prefix, key), np.max(value))
        logger.scalar_summary('{0}debug_extended/{1}_min'.format(prefix, key), np.min(value))

    return


def _shell_eps_log(logger, agent, agent_idx, itr_counter, task_counter, dict_logs):
    logger.info(Fore.BLUE + 'agent %d, task %d / episode %d, total steps %d ' \
    'mean/max/min reward %f/%f/%f' % (agent_idx, task_counter, \
        itr_counter,
        agent.total_steps,
        np.mean(agent.episode_returns),
        np.max(agent.episode_returns),
        np.min(agent.episode_returns)
    ))
    #logger.scalar_summary('agent_{0}/last_episode_avg_reward'.format(agent_idx), \
    #    np.mean(agent.last_episode_rewards))
    #logger.scalar_summary('agent_{0}/last_episode_std_reward'.format(agent_idx), \
    #    np.std(agent.last_episode_rewards))
    #logger.scalar_summary('agent_{0}/last_episode_max_reward'.format(agent_idx), \
    #    np.max(agent.last_episode_rewards))
    #logger.scalar_summary('agent_{0}/last_episode_min_reward'.format(agent_idx), \
    #    np.min(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/episode_avg_reward'.format(agent_idx), \
        np.mean(agent.episode_returns))
    logger.scalar_summary('agent_{0}/episode_std_reward'.format(agent_idx), \
        np.std(agent.episode_returns))
    logger.scalar_summary('agent_{0}/episode_max_reward'.format(agent_idx), \
        np.max(agent.episode_returns))
    logger.scalar_summary('agent_{0}/episode_min_reward'.format(agent_idx), \
        np.min(agent.episode_returns))

    prefix = 'agent_{0}_'.format(agent_idx)
    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('{0}debug/{1}_avg_norm'.format(prefix, tag), np.mean(value_norm))
            logger.scalar_summary('{0}debug/{1}_avg'.format(prefix, tag), value.mean())
            logger.scalar_summary('{0}debug/{1}_std'.format(prefix, tag), value.std())
            logger.scalar_summary('{0}debug/{1}_max'.format(prefix, tag), value.max())
            logger.scalar_summary('{0}debug/{1}_min'.format(prefix, tag), value.min())

    '''for key, value in dict_logs.items():
        logger.scalar_summary('{0}debug_extended/{1}_avg'.format(prefix, key), np.mean(value))
        logger.scalar_summary('{0}debug_extended/{1}_std'.format(prefix, key), np.std(value))
        logger.scalar_summary('{0}debug_extended/{1}_max'.format(prefix, key), np.max(value))
        logger.scalar_summary('{0}debug_extended/{1}_min'.format(prefix, key), np.min(value))'''

    return


# Deprecated trainers. Please use the concurrent versions further down in this file.
'''
shell training: single processing, all agents are executed/trained in a single process
with each agent taking turns to execute a training step.

NOTE: task boundaries are explictly given to the agents. this means that each agent
knows when task changes. 
'''
def shell_train(agents, logger):
    num_agents = len(agents)
    shell_done = [False,] * num_agents
    shell_iterations = [0,] * num_agents
    shell_tasks = [agent.config.cl_tasks_info for agent in agents] # tasks for each agent
    shell_task_ids = [agent.config.task_ids for agent in agents]
    shell_task_counter = [0,] * num_agents

    shell_eval_tracker = [False,] * num_agents
    shell_eval_data = []
    num_eval_tasks = len(agents[0].evaluation_env.get_all_tasks())
    shell_eval_data.append(np.zeros((num_agents, num_eval_tasks), dtype=np.float32))
    shell_metric_icr = [] # icr => instant cumulative reward metric
    eval_data_fhs = [open(logger.log_dir + '/eval_metrics_agent_{0}.csv'.format(agent_idx), 'a', \
        buffering=1) for agent_idx in range(num_agents)]

    if agents[0].task.name == agents[0].config.ENV_METAWORLD or \
        agents[0].task.name == agents[0].config.ENV_CONTINUALWORLD:
        itr_log_fn = _shell_itr_log_mw
    else:
        itr_log_fn = _shell_itr_log

    #print()
    logger.info(Fore.BLUE + '*****start shell training')

    # set the first task each agent is meant to train on
    for agent_idx, agent in enumerate(agents):
        states_ = agent.task.reset_task(shell_tasks[agent_idx][0])
        agent.states = agent.config.state_normalizer(states_)
        logger.info(Fore.BLUE + '*****agent {0} /setting first task (task 0)'.format(agent_idx))
        logger.info(Fore.BLUE + 'task: {0}'.format(shell_tasks[agent_idx][0]['task']))
        logger.info(Fore.BLUE + 'task_label: {0}'.format(shell_tasks[agent_idx][0]['task_label']))
        agent.task_train_start(shell_tasks[agent_idx][0]['task_label'])
        #print()
    del states_

    while True:
        for agent_idx, agent in enumerate(agents):
            if shell_done[agent_idx]:
                continue
            dict_logs = agent.iteration()
            shell_iterations[agent_idx] += 1
            # tensorboard log
            if shell_iterations[agent_idx] % agent.config.iteration_log_interval == 0:
                itr_log_fn(logger, agent, agent_idx, shell_iterations[agent_idx], \
                    shell_task_counter[agent_idx], dict_logs)

            # evaluation block
            if (agent.config.eval_interval is not None and \
                shell_iterations[agent_idx] % agent.config.eval_interval == 0):
                logger.info(Fore.BLUE + '*****agent {0} / evaluation block'.format(agent_idx))
                _task_ids = shell_task_ids[agent_idx]
                _tasks = shell_tasks[agent_idx]
                _names = [eval_task_info['name'] for eval_task_info in _tasks]
                logger.info(Fore.BLUE + 'eval tasks: {0}'.format(', '.join(_names)))
                for eval_task_idx, eval_task_info in zip(_task_ids, _tasks):
                    agent.task_eval_start(eval_task_info['task_label'])
                    eval_states = agent.evaluation_env.reset_task(eval_task_info)
                    agent.evaluation_states = eval_states
                    # performance (perf) can be success rate in (meta-)continualworld or
                    # rewards in other environments
                    perf, eps = agent.evaluate_cl(num_iterations=agent.config.evaluation_episodes)
                    agent.task_eval_end()
                    shell_eval_data[-1][agent_idx, eval_task_idx] = np.mean(perf)
                shell_eval_tracker[agent_idx] = True
                # save latest eval data for current agent to csv
                _record = np.concatenate([shell_eval_data[-1][agent_idx, : ], \
                    np.array(time.time()).reshape(1,)])
                np.savetxt(eval_data_fhs[agent_idx], _record.reshape(1, -1),delimiter=',',fmt='%.4f')
                del _record

            # checker for end of task training
            if not agent.config.max_steps:
                raise ValueError(Fore.WHITE + '`max_steps` should be set for each agent')
            task_steps_limit = agent.config.max_steps[shell_task_counter[agent_idx]] * \
                (shell_task_counter[agent_idx] + 1)
            if agent.total_steps >= task_steps_limit:
                #print()
                task_counter_ = shell_task_counter[agent_idx]
                logger.info(Fore.BLUE + '*****agent {0} / end of training on task {1}'.format(agent_idx, \
                    task_counter_))
                agent.task_train_end()

                task_counter_ += 1
                shell_task_counter[agent_idx] = task_counter_
                if task_counter_ < len(shell_tasks[agent_idx]):
                    logger.info(Fore.BLUE + '*****agent {0} / set next task {1}'.format(agent_idx, task_counter_))
                    logger.info(Fore.BLUE + 'task: {0}'.format(shell_tasks[agent_idx][task_counter_]['task']))
                    logger.info(Fore.BLUE + 'task_label: {0}'.format(shell_tasks[agent_idx][task_counter_]['task_label']))
                    states_ = agent.task.reset_task(shell_tasks[agent_idx][task_counter_]) # set new task
                    agent.states = agent.config.state_normalizer(states_)
                    agent.task_train_start(shell_tasks[agent_idx][task_counter_]['task_label'])

                    # ping other agents to see if they have knoweledge (mask) of current task
                    logger.info(Fore.BLUE + '*****agent {0} / pinging other agents to leverage on existing ' \
                        'knowledge about task'.format(agent_idx))
                    a_idxs = list(range(len(agents)))
                    a_idxs.remove(agent_idx)
                    other_agents = [agents[i] for i in a_idxs]
                    found_knowledge = agent.ping_agents(other_agents)
                    if found_knowledge > 0:
                        logger.info(Fore.BLUE + 'found knowledge about task from other agents')
                        logger.info(Fore.BLUE + 'number of task knowledge found: {0}'.format(found_knowledge))
                    else:
                        logger.info(Fore.BLUE + 'could not find any agent with knowledge about task')
                    del states_
                    #print()
                else:
                    shell_done[agent_idx] = True # training done for all task for agent
                    logger.info(Fore.BLUE + '*****agent {0} / end of all training'.format(agent_idx))
                del task_counter_
                    
        if all(shell_eval_tracker):
            _metrics = shell_eval_data[-1]
            # compute icr
            _max_reward = _metrics.max(axis=0) 
            _agent_ids = _metrics.argmax(axis=0).tolist()
            _agent_ids = ', '.join([str(_agent_id) for _agent_id in _agent_ids])
            icr = _max_reward.sum()
            shell_metric_icr.append(icr)
            # log eval to file/screen and tensorboard
            logger.info(Fore.BLUE + '*****shell evaluation:')
            logger.info(Fore.BLUE + 'best agent per task:'.format(_agent_ids))
            logger.info(Fore.BLUE + 'shell eval ICR: {0}'.format(icr))
            logger.info(Fore.BLUE + 'shell eval TP: {0}'.format(np.sum(shell_metric_icr)))
            logger.scalar_summary('shell_eval/icr', icr)
            logger.scalar_summary('shell_eval/tpot', np.sum(shell_metric_icr))
            # reset eval tracker
            shell_eval_tracker = [False for _ in shell_eval_tracker]
            # initialise new eval block
            shell_eval_data.append(np.zeros((num_agents, num_eval_tasks), dtype=np.float32))

        if all(shell_done):
            break

    for i in range(len(eval_data_fhs)):
        eval_data_fhs[i].close()
    # discard last eval data entry as it was not used.
    if np.all(shell_eval_data[-1] == 0.):
        shell_eval_data.pop(-1)
    # save eval metrics
    to_save = np.stack(shell_eval_data, axis=0)
    with open(logger.log_dir + '/eval_metrics.npy', 'wb') as f:
        np.save(f, to_save)

    for agent in agents:
        agent.close()
    return

'''
shell training: concurrent processing, all agents are executed/trained in across multiple
processes (either in a single machine, or distributed machines)

NOTE: task boundaries are explictly given to the agents. this means that each agent
knows when task changes. 
'''
def shell_dist_train(agent, comm, agent_id, num_agents):
    logger = agent.config.logger
    #print()
    logger.info(Fore.BLUE + '*****start shell training')

    shell_done = False
    shell_iterations = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    shell_task_ids = agent.config.task_ids
    shell_task_counter = 0


    knowledge_base = dict()


    shell_eval_tracker = False
    shell_eval_data = []
    num_eval_tasks = len(agent.evaluation_env.get_all_tasks())
    shell_eval_data.append(np.zeros((num_eval_tasks, ), dtype=np.float32))
    shell_metric_icr = [] # icr => instant cumulative reward metric. NOTE may be redundant now
    eval_data_fh = open(logger.log_dir + '/eval_metrics_agent_{0}.csv'.format(agent_id), 'a', \
        buffering=1) # buffering=1 means flush data to file after every line written
    shell_eval_end_time = None

    if agent.task.name == agent.config.ENV_METAWORLD or \
        agent.task.name == agent.config.ENV_CONTINUALWORLD:
        itr_log_fn = _shell_itr_log_mw
    else:
        itr_log_fn = _shell_itr_log

    await_response = [False,] * num_agents # flag
    # set the first task each agent is meant to train on
    states_ = agent.task.reset_task(shell_tasks[0])
    agent.states = agent.config.state_normalizer(states_)
    logger.info(Fore.BLUE + '*****agent {0} / setting first task (task 0)'.format(agent_id))
    logger.info(Fore.BLUE + 'task: {0}'.format(shell_tasks[0]['task']))
    logger.info(Fore.BLUE + 'task_label: {0}'.format(shell_tasks[0]['task_label']))
    agent.task_train_start(shell_tasks[0]['task_label'])
    #print()
    del states_

    msg = None
    while True:
        #print(await_response)
        # listen for and send info (task label or NULL message) to other agents
        other_agents_request = comm.send_receive_request(msg)
        #print(other_agents_request)
        msg = None # reset message

        requests = []
        for req in other_agents_request:
            #print(req)
            if req is None: continue
            req['mask'] = agent.embedding_to_mask(req['task_label'])
            requests.append(req)
        if len(requests) > 0:
            comm.send_response(requests)

        # if the agent earlier sent a request, check whether response has been sent.
        if any(await_response):
            logger.info(Fore.BLUE + 'awaiting response: {0}'.format(await_response))
            masks = []
            received_masks = comm.receive_response()
            for i in range(len(await_response)):
                if await_response[i] is False: continue

                if received_masks[i] is False: continue
                elif received_masks[i] is None: await_response[i] = False
                else:
                    masks.append(received_masks[i])
                    await_response[i] = False
            logger.info(Fore.BLUE + 'number of task knowledge received: {0}'.format(len(masks)))
            # TODO still need to fix how mask is used.
            agent.distil_task_knowledge(masks)
                    
        # agent iteration (training step): collect on policy data and optimise agent
        dict_logs = agent.iteration()
        shell_iterations += 1

        # tensorboard log
        if shell_iterations % agent.config.iteration_log_interval == 0:
            itr_log_fn(logger, agent, agent_id, shell_iterations, shell_task_counter, dict_logs)
            
            # Create a dictionary to store the most recent iteration rewards for a mask. Update in every iteration
            # logging cycle. Take average of all worker averages as the most recent reward score for a given task
            knowledge_base[np.array2string(shell_tasks[shell_task_counter]['task_label'], precision=2, separator=', ', suppress_small=True)] = np.mean(agent.iteration_rewards)
            #print(knowledge_base)


        # evaluation block
        if (agent.config.eval_interval is not None and \
            shell_iterations % agent.config.eval_interval == 0):
            logger.info(Fore.BLUE + '*****agent {0} / evaluation block'.format(agent_id))
            _task_ids = shell_task_ids
            _tasks = shell_tasks
            _names = [eval_task_info['name'] for eval_task_info in _tasks]
            logger.info(Fore.BLUE + 'eval tasks: {0}'.format(', '.join(_names)))
            for eval_task_idx, eval_task_info in zip(_task_ids, _tasks):
                agent.task_eval_start(eval_task_info['task_label'])
                eval_states = agent.evaluation_env.reset_task(eval_task_info)
                agent.evaluation_states = eval_states
                # performance (perf) can be success rate in (meta-)continualworld or
                # rewards in other environments
                perf, eps = agent.evaluate_cl(num_iterations=agent.config.evaluation_episodes)
                agent.task_eval_end()
                shell_eval_data[-1][eval_task_idx] = np.mean(perf)
            shell_eval_tracker = True
            shell_eval_end_time = time.time()

        # end of current task training. move onto next task or end training if last task.
        if not agent.config.max_steps: raise ValueError('`max_steps` should be set for each agent')
        task_steps_limit = agent.config.max_steps[shell_task_counter] * (shell_task_counter + 1)
        if agent.total_steps >= task_steps_limit:
            #print()
            task_counter_ = shell_task_counter
            logger.info(Fore.BLUE + '*****agent {0} / end of training on task {1}'.format(agent_id, task_counter_))
            agent.task_train_end()

            task_counter_ += 1
            shell_task_counter = task_counter_
            if task_counter_ < len(shell_tasks):
                # new task
                logger.info(Fore.BLUE + '*****agent {0} / set next task {1}'.format(agent_id, task_counter_))
                logger.info(Fore.BLUE + 'task: {0}'.format(shell_tasks[task_counter_]['task']))
                logger.info(Fore.BLUE + 'task_label: {0}'.format(shell_tasks[task_counter_]['task_label']))
                states_ = agent.task.reset_task(shell_tasks[task_counter_]) # set new task
                agent.states = agent.config.state_normalizer(states_)
                agent.task_train_start(shell_tasks[task_counter_]['task_label'])

                # set message (task_label) that will be sent to other agent as a request for
                # task knowledge (mask) about current task. this will be sent in the next
                # receive/send request cycle.
                logger.info(Fore.BLUE + '*****agent {0} / query other agents using current task label'\
                    .format(agent_id))
                msg = shell_tasks[task_counter_]['task_label']


                



                await_response = [True,] * num_agents
                del states_
                #print()
            else:
                shell_done = True # training done for all task for agent
                logger.info(Fore.BLUE + '*****agent {0} / end of all training'.format(agent_id))
            del task_counter_
                    
        if shell_eval_tracker:
            # log the last eval metrics to file
            _record = np.concatenate([shell_eval_data[-1],np.array(shell_eval_end_time).reshape(1,)])
            np.savetxt(eval_data_fh, _record.reshape(1, -1), delimiter=',', fmt='%.4f')
            del _record

            # reset eval tracker and add new buffer to save next eval metrics
            shell_eval_tracker = False
            shell_eval_data.append(np.zeros((num_eval_tasks, ), dtype=np.float32))
        #if all(shell_eval_tracker):
        #    _metrics = shell_eval_data[-1]
        #    # compute icr
        #    _max_reward = _metrics.max(axis=0) 
        #    _agent_ids = _metrics.argmax(axis=0).tolist()
        #    _agent_ids = ', '.join([str(_agent_id) for _agent_id in _agent_ids])
        #    icr = _max_reward.sum()
        #    shell_metric_icr.append(icr)
        #    # log eval to file/screen and tensorboard
        #    logger.info('*****shell evaluation:')
        #    logger.info('best agent per task:'.format(_agent_ids))
        #    logger.info('shell eval ICR: {0}'.format(icr))
        #    logger.info('shell eval TP: {0}'.format(np.sum(shell_metric_icr)))
        #    logger.scalar_summary('shell_eval/icr', icr)
        #    logger.scalar_summary('shell_eval/tpot', np.sum(shell_metric_icr))
        #    # reset eval tracker
        #    shell_eval_tracker = [False for _ in shell_eval_tracker]
        #    # initialise new eval block
        #    shell_eval_data.append(np.zeros((1, num_eval_tasks), dtype=np.float32))


        #print(await_response)

        if shell_done:
            break
        comm.barrier()
    # end of while True

    #eval_data_fh.close()
    # discard last eval data entry as it was not used.
    #if np.all(shell_eval_data[-1] == 0.):
    #    shell_eval_data.pop(-1)
    # save eval metrics
    #to_save = np.stack(shell_eval_data, axis=0)
    #with open(logger.log_dir + '/eval_metrics_agent_{0}.npy'.format(agent_id), 'wb') as f:
    #    np.save(f, to_save)

    agent.close()
    return
    


# Concurrent implementations
'''
shell training: concurrent processing for event-based communication. a multitude of improvements have been made compared
to the previous shell_dist_train.
'''
#@profile
def trainer_learner(agent, comm, agent_id, manager, mask_interval, mode):
    ###############################################################################
    ### Setup logger
    logger = agent.config.logger
    print(Fore.WHITE, end='') 
    logger.info('***** start l2d2-c training')


    ###############################################################################
    ### Setup trainer loop pre-requisites
    shell_done = False
    shell_iterations = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    #shell_task_ids = agent.config.task_ids
    shell_task_counter = 0
    #shell_eval_tracker = False
    shell_eval_data = []
    num_eval_tasks = len(agent.evaluation_env.get_all_tasks())
    shell_eval_data.append(np.zeros((num_eval_tasks, ), dtype=np.float32))
    #shell_metric_icr = [] # icr => instant cumulative reward metric. NOTE may be redundant now
    #eval_data_fh = open(logger.log_dir + '/eval_metrics_agent_{0}.csv'.format(agent_id), 'a', 
    #    buffering=1) # buffering=1 means flush data to file after every line written
    #shell_eval_end_time = None

    idling = True   # Flag to handle idling behaviour when curriculum ends.
    dict_to_query = None      # Variable to store the current task dictionary recrod to query for.


    ###############################################################################
    ### Select iteration logging function based on environment. Required for Meta World and Continual World
    if agent.task.name == agent.config.ENV_METAWORLD or agent.task.name == agent.config.ENV_CONTINUALWORLD or agent.task.name == agent.config.ENV_COMPOSUITE:
        itr_log_fn = _shell_itr_log_mw

    else:
        itr_log_fn = _shell_itr_log


    ###############################################################################
    ### Set the first task each agent is meant to train on
    states_ = agent.task.reset_task(shell_tasks[0])
    agent.states = agent.config.state_normalizer(states_)
    logger.info(f'***** ENVIRONMENT SWITCHING TASKS')
    logger.info(f'***** agent {agent_id} / setting first task (task 0)')
    logger.info(f"***** task: {shell_tasks[0]['task']}")
    logger.info(f"***** task_label: {shell_tasks[0]['task_label']}")

    # Set first task mask and record manually otherwise we run into issues with the implementation in the model.
    #agent.task_train_start_emb(task_embedding=None)

    # NOTE: ADDED detect.add_embedding() to accomodate the WEIGHTED AVG COSINE SIM
    agent.current_task_emb = torch.zeros(agent.get_task_emb_size())
    agent.task_train_start_emb(task_embedding=agent.current_task_emb, current_reward=agent.iteration_rewards)   # TODO: There is an issue with this which is that the first task will be set as zero and then the detect module with do some learning, find that the task does not match the zero embedding and start another task change. This leaves the first entry to a task change as useless. Also issues if we try to moving average this
    #agent.detect.add_embedding(agent.current_task_emb, np.mean(agent.iteration_rewards))
    del states_


    ###############################################################################
    ### Start the comm module with the initial states and the first task label.
    # Returns shared queues to enable interaction between comm and agent.
    queue_label, queue_mask, queue_label_send, queue_mask_recv = comm.parallel(manager)


    ###############################################################################
    ### Logging setup (continued)
    tb_writer_emb = SummaryWriter(logger.log_dir + '/Detect_Component_Generated_Embeddings')
    _embeddings, _labels, exchanges, task_times, detect_module_activations = [], [], [], [], []
    task_times.append([0, shell_iterations, np.argmax(shell_tasks[0]['task_label'], axis=0), time.time()])
    detect_activations_log_path = logger.log_dir + '/detect_activations.csv'
    masks_log_path = logger.log_dir + '/exchanges.csv'
    emb_dist_log = logger.log_dir + '/distances.csv'
    m_dist_log1 = logger.log_dir + '/maha_cov_ident.csv'
    m_dist_log2 = logger.log_dir + '/maha_cov_mean.csv'
    cossim_log = logger.log_dir + '/cos_sim.csv'
    density_log = logger.log_dir + '/density.csv'
    emd_log = logger.log_dir + '/emd.csv'
    wdist_log = logger.log_dir + '/wdist_log.csv'


    '''###############################################################################
    ### Comm module event handlers. These run in parallel to enable the interactions between the comm and agent.
    def mask_handler():
        """
        Handles incoming masks from other agents. Linearly combines masks and adds resulting mask to network.
        """
        while True:
            masks_list  = queue_mask.get()
            
            logger.info(Fore.WHITE + f'\n######### MASK RECEIVED FROM COMM #########')

            _masks = []
            _avg_embeddings = []
            _avg_rewards = []
            _mask_labels = []
            print(f'masks list length: {len(masks_list)}')

            try:
                if len(masks_list) > 0:
                    for mask_response_dict in masks_list:

                        mask = mask_response_dict['mask']
                        embedding = mask_response_dict['embedding']
                        reward = mask_response_dict['reward']
                        label = mask_response_dict['label']
                        ip = mask_response_dict['ip']
                        port = mask_response_dict['port']

                        print(type(label), len(label), label)

                        #_masks.append(mask)
                        _masks.append(agent.vec_to_mask(mask.to(agent.config.DEVICE))) # Use this one if using unified LC
                        _avg_embeddings.append(embedding)
                        _avg_rewards.append(reward)
                        _mask_labels.append(label)

                        # Log successful mask transfer
                        data = [
                            {
                                'iteration': shell_iterations,
                                'ip': ip,
                                'port': port,
                                'task_id': np.argmax(label,axis=0),
                                'reward': reward,
                                'embedding': embedding,
                                'mask_dim': len(mask),
                                'mask_tensor': mask
                            }
                        ]
                    
                        df = pd.DataFrame(data)
                        df.to_csv(masks_log_path, mode='a', header=not pd.io.common.file_exists(masks_log_path), index=False)


                        #exchanges.append([shell_iterations, ip, port, np.argmax(label, axis=0), reward, embedding, len(mask), mask])
                        #np.savetxt(logger.log_dir + '/exchanges_{0}.csv'.format(agent_id), exchanges, delimiter=',', fmt='%s')
                    
                    #logger.info(Fore.WHITE + f'Updating seen tasks dictionary with new data')
                    # Update the knowledge base with the expected reward
                    #agent.update_seen_tasks(_avg_embeddings[0], _avg_rewards[0], _mask_labels[0])#knowledge_base.update({tuple(label.tolist()): reward})
                    
                    # Traceback (most recent call last):
                    # File "/home/lunet/cosn2/detect-l2d2c/deeprl-shell/deep_rl/utils/trainer_shell.py", line 671, in mask_handler
                    #     agent.update_seen_tasks(_avg_embeddings[0], _avg_rewards[0], _mask_labels[0])
                    # IndexError: list index out of range

                    logger.info(Fore.WHITE + f'COMPOSING RECEIVED MASKS')
                    # Update the network with the linearly combined mask
                    #agent.distil_task_knowledge_embedding(_masks[0])       # This will only take the first mask in the list
                    #agent.consolidate_incoming(_masks)                      # This will take all the masks in the list and linearly combine with the random/current mask
                    agent.update_community_masks(_masks, np.mean(agent.iteration_rewards))
                    _masks = []

                    logger.info(Fore.WHITE + 'COMPOSED MASK ADDED TO NETWORK!')
            except Exception as e:
                traceback.print_exc()

    def conv_handler():
        """
        Handles interval label to mask conversions for outgoing mask responses.
        """
        while True:
            try:
                to_convert = queue_label_send.get()

                logger.info(Fore.WHITE + 'GOT ID TO CONVERT TO MASK')
                logger.info(f"MASK/TASK ID: {to_convert['sender_task_id']}")

                sender_task_id = to_convert['sender_task_id']
                mask = agent.idx_to_mask(sender_task_id)

                print(sender_task_id)
                print(agent.seen_tasks[sender_task_id])

                reward = agent.seen_tasks[sender_task_id]['reward']
                emb = agent.seen_tasks[sender_task_id]['task_emb']
                label = agent.seen_tasks[sender_task_id]['ground_truth']
                print(Fore.LIGHTRED_EX + f'Found valid mask: {mask} with reward: {reward} and emb: {emb}')

                to_convert['response_mask'] = mask
                to_convert['response_reward'] = reward
                to_convert['response_embedding'] = emb
                to_convert['response_label'] = label
                queue_mask_recv.put((to_convert))
            except Exception as e:
                traceback.print_exc()
    

    ###############################################################################
    ### Start threads for the mask and conversion handlers.
    t_mask = mpd.Pool(processes=1)
    t_conv = mpd.Pool(processes=1)
    t_mask.apply_async(mask_handler)
    t_conv.apply_async(conv_handler)'''
    

    ###############################################################################
    ### MAIN OPERATIONAL LOOP
    while True:
        start_time = time.time()
        ###############################################################################
        ### Idling behaviour. Idles until terminated when curriculum is completed.
        # While idling the agent acts as a server that can be queried for knowledge by
        # other agents.
        if shell_done:
            if idling:
                print('Agent is idling...') # Once idling the agent acts as a server that can be queried for knowledge until the agent encounters a new task (support not implemented yet)
                
                # Log all the embeddings and labels to tensorboard projector
                emb_t = torch.stack(tuple(_embeddings))
                tb_writer_emb.add_embedding(emb_t, metadata=_labels, global_step=shell_iterations)
                
                idling = False
                # Alternatively we can shutdown the agent here or do something for the experiment termination.
                
            #if omniscient_mode:
            #    if shell_iterations % mask_interval == 0:
            #        queue_label.put(None)

            #    shell_iterations += 1

            time.sleep(2) # Sleep value ensures the idling works as intended (fixes a problem encountered with the multiprocessing)
            continue
        print()


        
        ###############################################################################
        ### Registry logging output.
        logger.info(Fore.RED + 'GLOBAL REGISTRY (seen_tasks dict)')
        for key, val in agent.seen_tasks.items(): logger.info(f"{key} --> embedding: {val['task_emb']}, reward: {val['reward']}, ground truth task id: {np.argmax(val['ground_truth'], axis=0)}, label length: {len(val['ground_truth'])}")

        ###############################################################################
        ### Query for knowledge using communication process. Send label/embedding to the communication module to query for relevant knowledge from other peers.
        if dict_to_query is not None:
            if shell_iterations % mask_interval == 0:
                # Approach 2: At this point consolidate masks and then we can reset beta parameters. Then we can get new masks from network and combine.
                dict_to_query['shell_iteration'] = shell_iterations
                queue_label.put(dict_to_query)

        # Report performance to evaluation agent if present. Otherwise skip.
        print(agent.config.evaluator_present.value)
        if agent.config.evaluator_present.value == True:
            logger.info('Reporting performance to evaluation agent')
            dict_to_report = agent.seen_tasks[agent.current_task_key]
            dict_to_report['shell_iteration'] = shell_iterations
            dict_to_report['mask'] = agent.idx_to_mask(agent.current_task_key)
            dict_to_report['eval'] = True
            dict_to_report['parameters'] = None
            queue_label.put(dict_to_report)


        
        ###############################################################################
        ### Agent training iteration: collect on policy data and optimise the agent
        '''
        Handles the data collection and optimisation within the agent.
        TODO: Look into multihreading/multiprocessing the data collection and optimisation of the agent
                to achieve the continous data collection we want for a real world scenario.
            
            Possibly the optimisation could be made a seperate process parallel to the data collection
                process, similar to the communication-agent(trainer) architecture. Data collection and the code
                below would run together in the main loop.

        NOTE: Parallelising the backward and forward passes may not be possible using CUDA due to limitations
                on CUDA parallelisation. It could be done if we train the system on CPU using
                mulithreading or multiprocessing however it is challenging as there needs to be some sort of
                synchronisation.
        '''
        dict_logs = agent.iteration()
        shell_iterations += 1

        # Log the beta parameters for the curren task
        agent.log_betas(shell_iterations)
        

        
        ###############################################################################
        ### Run detect module. Generates embedding for SAR. Perform check to see if there has been a task change or not.
        _dist_threshold = agent.emb_dist_threshold
        if shell_iterations != 0 and shell_iterations % agent.detect_module_activation_frequency == 0 and agent.data_buffer.size() >= (agent.detect.get_num_samples()):
            # Run the detect module on SAR and return some logging output.
            task_change_flag, new_emb, ground_truth_task_label, dist_arr, emb_bool, agent_seen_tasks = run_detect_module(agent)

            emb_dist = dist_arr[0]
            m_dist1 = dist_arr[1]
            m_dist2 = dist_arr[2]
            cos_sim = dist_arr[3]
            density = dist_arr[4]
            emd = dist_arr[5]
            w_dist = dist_arr[6]

            # Log euclidean distance with moving average on current embedding
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(emb_dist)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(emb_dist_log, mode='a', header=not pd.io.common.file_exists(emb_dist_log), index=False)

            # Log mahalanobis distance with moving average with identity covariance matrix
            '''data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(m_dist1)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(m_dist_log1, mode='a', header=not pd.io.common.file_exists(m_dist_log1), index=False)'''

            # Log mahalanobis distance with moving average with mean covariance matrix
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(m_dist2)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(m_dist_log2, mode='a', header=not pd.io.common.file_exists(m_dist_log2), index=False)

            # Log cosine similarity with moving average
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(cos_sim)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(cossim_log, mode='a', header=not pd.io.common.file_exists(cossim_log), index=False)

            # Kernel density with moving average
            '''data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(density)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(density_log, mode='a', header=not pd.io.common.file_exists(density_log), index=False)'''

            # Wasserstein distance / Earth Mover's Distance
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(emd)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(emd_log, mode='a', header=not pd.io.common.file_exists(emd_log), index=False)

            # Wasserstein distance reference
            '''data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(w_dist)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(wdist_log, mode='a', header=not pd.io.common.file_exists(wdist_log), index=False)'''
            
            if task_change_flag:
                logger.info(Fore.YELLOW + f'TASK CHANGE DETECTED! NEW MASK CREATED. CURRENT TASK INDEX: {agent.current_task_key}')
            
                #log_string = f'Time: {time.time()}, Iteration: {shell_iterations}, Num samples for detection: {agent.detect.get_num_samples()}, Task change flag: {task_change_flag}, New embedding: {new_emb}, Ground truth label: {ground_truth_task_label}, Current embedding: {agent.current_task_emb}, Threshold: {_dist_threshold}, Distance: {emb_dist}, Embedding similarity: {emb_bool}, Agent seen tasks: {agent_seen_tasks}'
                #detect_module_activations.append([log_string])
                #np.savetxt(logger.log_dir + '/detect_activations_{0}.csv'.format(agent_id), detect_module_activations, delimiter=',', fmt='%s')

                data = [
                    {
                        'Iteration': shell_iterations,
                        'Time': time.time(),
                        'Num samples for detection': agent.detect.get_num_samples(),
                        'New embedding': new_emb,
                        'Ground truth label': ground_truth_task_label,
                        'Current embedding': agent.current_task_emb,
                        'Threshold': _dist_threshold,
                        'Distance': emb_dist,
                        'Similar': emb_bool,
                        'Agent seen_tasks()': agent_seen_tasks 
                    }
                ]
                df = pd.DataFrame(data)
                df.to_csv(detect_activations_log_path, mode='a', header=not pd.io.common.file_exists(detect_activations_log_path), index=False)
            
            # Update the dictionary containing the current task embedding to query for.
            dict_to_query = agent.seen_tasks[agent.current_task_key]
            dict_to_query['parameters'] = 0.5 #€ Cosine similarity threshold

            # Logging embeddings and labels
            if new_emb is not None:
                _label_one_hot = torch.tensor(np.array([ground_truth_task_label]))
                
                # Convert one-hot label to integer
                _label = torch.argmax(_label_one_hot).item()

                _embeddings.append(new_emb)
                _labels.append(_label)

                logger.info(Fore.WHITE + f'Embedding: {new_emb}')
                logger.info(f'Task ID: {_label}')
                logger.info(f'Distance: {emb_dist}')
                logger.info(f'Threshold: {agent.emb_dist_threshold}')
                #emb_t = torch.stack(tuple(_embeddings))
                #l_t = torch.stack(tuple(_labels))
                #tb_writer_emb.add_embedding(emb_t, metadata=_labels, global_step=shell_iterations)
        
            
        
        ###############################################################################
        ### Logs metrics to tensorboard log file and updates the embedding, reward pair in this cycle for a particular task.
        if shell_iterations % agent.config.iteration_log_interval == 0:
            itr_log_fn(logger, agent, agent_id, shell_iterations, shell_task_counter, dict_logs)
                
            # Save agent model
            agent.save(agent.config.log_dir + '/%s-%s-model-%s.bin' % (agent.config.agent_name, agent.config.tag, agent.task.name))
        


        
        ###############################################################################
        ### Environment task change at the end of the max steps for each task. Agent is not aware of this change and must detect it using the detect module.
        '''
        # end of current task training. move onto next task or end training if last task.
        # i.e., Task Change occurs here. For detect module, if the task embedding signifies a task
        # change then that should occur here.

        If we want to use a Fetch All mode for ShELL then we need to add a commmunication component
        at task change which broadcasts the mask to all other agents currently on the network.

        Otherwise the current implementation is a On Demand mode where each agent requests knowledge
        only when required.
        '''
        if not agent.config.max_steps: raise ValueError('`max_steps` should be set for each agent')
        task_steps_limit = agent.config.max_steps[shell_task_counter] * (shell_task_counter + 1)

        # If agent completes the maximum number of steps for a task then switch to the next task in the curriculum.
        if agent.total_steps >= task_steps_limit:
            task_counter_ = shell_task_counter
            logger.info('\n' + Fore.WHITE + f'*****agent {agent_id} / end of training on task {task_counter_}')
            
            #MOVED to Assing EMB in PPO agent.task_train_end()

            # Increment task counter
            task_counter_ += 1
            shell_task_counter = task_counter_

            # If curriculum is not completed, switch to the next task in the curriculum
            if task_counter_ < len(shell_tasks):
                # new task
                logger.info(Fore.WHITE + f'***** ENVIRONMENT SWITCHING TASKS')
                logger.info(Fore.WHITE + f'***** agent {agent_id} / set next task {task_counter_}')
                logger.info(Fore.WHITE + f"***** task: {shell_tasks[task_counter_]['task']}")
                logger.info(Fore.WHITE + f"***** task_label: {shell_tasks[task_counter_]['task_label']}")
                
                # Set the new task from the environment. Agent remains unaware of this change and will continue until
                # detect module detects distrubtion shift.
                states_ = agent.task.reset_task(shell_tasks[task_counter_]) # reset_task sets the new task and returns the reset intial states.
                agent.states = agent.config.state_normalizer(states_)
                
                #MOVED to Assing EMB in PPO agent agent.task_train_start(shell_tasks[task_counter_]['task_label'])

                del states_

                task_times.append([task_counter_, shell_iterations, np.argmax(shell_tasks[task_counter_]['task_label'], axis=0), time.time()])
                np.savetxt(logger.log_dir + '/task_changes_{0}.csv'.format(agent_id), task_times, delimiter=',', fmt='%s')

            else:
                shell_done = True # training done for all task for agent. This leads to the idling behaviour in next iteration.
                logger.info(f'*****agent {agent_id} / end of all training')

            del task_counter_
        


        logger.info(f'{Fore.BLUE}----------------------- Iteration complete in {time.time() - start_time} seconds -----------------------\n')


'''
shell evaluation: concurrent processing for event-based communication.
'''
def trainer_evaluator(agent, comm, agent_id, manager, knowledge_base):
    logger = agent.config.logger
    logger.info(Fore.BLUE + '*****start shell training')

    #Create a SummaryWriter object for the Evaluation Agent
    tb_writer = SummaryWriter(logger.log_dir + '/DMIU-ShELL_metrics')

    shell_done = False
    shell_iterations = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    shell_task_ids = agent.config.task_ids
    shell_task_counter = 0

    shell_agent_seed = agent.config.seed        # Chris
    agent.config.eval_interval = 1      # Manual override

    _task_ids = shell_task_ids
    _tasks = shell_tasks
    # agent.seen_tasks = {_tasks}
    agent.seen_tasks = {}
    for idx, eval_task_info in enumerate(_tasks):
        agent.seen_tasks[idx] = eval_task_info['task_label']
        knowledge_base[np.argmax(eval_task_info['task_label'], axis=0)] = 0.0
    _names = [eval_task_info['name'] for eval_task_info in _tasks]
    eval_task_info = zip(_task_ids, _tasks)

    for x, y in eval_task_info:
        print(x, y)

    shell_eval_tracker = False
    shell_eval_data = []
    num_eval_tasks = len(agent.evaluation_env.get_all_tasks())
    shell_eval_data.append(np.zeros((num_eval_tasks, ), dtype=np.float32))
    shell_metric_icr = [] # icr => instant cumulative reward metric. NOTE may be redundant now
    eval_data_fh = open(logger.log_dir + '/eval_metrics_agent_{0}.csv'.format(agent_id), 'a', \
        buffering=1) # buffering=1 means flush data to file after every line written
    shell_eval_end_time = None
    
    task_steps_limit = sum(agent.config.max_steps)

    # Msg can be embedding or task label.
    # Set msg to first task. The agents will then request knowledge on the first task.
    # this will ensure that other agents are aware that this agent is now working this task
    # until a task change happens.
    msg = _tasks[0]['task_label']

    queue_label, queue_mask, queue_label_send, queue_mask_recv = comm.parallel(manager)

    exchanges = []
    task_times = []
    detect_module_activations = [] #Chris
    task_times.append([0, shell_iterations, np.argmax(shell_tasks[0]['task_label'], axis=0), time.time()])

    masks_log_path = logger.log_dir + '/exchanges.csv'


    # Communication module interaction handlers
    def mask_handler():
        """
        Handles incoming masks from other agents. Distills the mask knowledge to the agent's network.
        """
        while True:
            ret  = queue_mask.get()
            mask = ret['mask']
            reward = ret['reward']
            label = ret['label']
            ip = ret['ip']
            port = ret['port']

            logger.info(Fore.WHITE + f'\nReceived mask: \nMask:{mask}\nReward:{reward}\nSrc:{ip, port}\nLabel:{label}')
            try:
                if mask is not None:
                    # Check if the mask reported reward is higher than that of the current mask. If true then distil knowledge
                    # Update the knowledge base with the expected reward
                    knowledge_base.update({np.argmax(label, axis=0): reward})
                    # Update the network with the mask
                    if agent.distil_task_knowledge_single(mask, label): logger.info(Fore.WHITE + 'KNOWLEDGE DISTILLED TO NETWORK!')
                    
                    data = [
                        {
                            'iteration': shell_iterations,
                            'ip': ip,
                            'port': port,
                            'task_id': np.argmax(label,axis=0),
                            'reward': reward,
                            'mask_dim': len(mask),
                            'mask_tensor': mask
                        }
                    ]
                    df = pd.DataFrame(data)
                    df.to_csv(masks_log_path, mode='a', header=not pd.io.common.file_exists(masks_log_path), index=False)
            
            except Exception as e:
                traceback.print_exc()

    t_mask = mpd.Pool(processes=1)
    t_mask.apply_async(mask_handler)

    for key, val in agent.seen_tasks.items(): print(f"{key, val}")
    for key, val in knowledge_base.items(): print(f"{key, val}")
    for idx, info in zip(_task_ids, _tasks): print(f"{idx, info}")

    #agent.network.eval()
    while True:
        #time.sleep(1)
        print()
        print(shell_iterations, shell_task_counter, agent.total_steps, agent.config.max_steps, task_steps_limit)
        agent.total_steps += agent.config.rollout_length * agent.config.num_workers

        # Increment iterations
        shell_iterations += 1
        logger.info(Fore.RED + 'GLOBAL REGISTRY (seen_tasks dict)')
        for key, val in agent.seen_tasks.items(): logger.info(f"{key, val}")

        logger.info(Fore.RED + 'GLOBAL KNOWLEDGE BASE (knowledge base)')
        for key, val in knowledge_base.items(): logger.info(f"{key, val}")

        logger.info(Fore.RED + 'TASKS')
        for key, val in zip(_task_ids, _tasks): logger.info(f"{key, val}")


        ###############################################################################
        ### EVALUATION BLOCK
        if (agent.config.eval_interval is not None and shell_iterations % agent.config.eval_interval == 0):
            logger.info(Fore.BLUE + '*****agent {0} / evaluation block'.format(agent_id))
            _task_ids = shell_task_ids
            _tasks = shell_tasks
            _names = [eval_task_info['name'] for eval_task_info in _tasks]
            logger.info(Fore.BLUE + 'eval tasks: {0}'.format(', '.join(_names)))

            for eval_task_idx, eval_task_info in zip(_task_ids, _tasks):
                agent.task_eval_start(eval_task_info['task_label'])     # agent.task_eval_start(np.array(eval_task_info['task_label']))
                eval_states = agent.evaluation_env.reset_task(eval_task_info)
                agent.evaluation_state = eval_states
                # performance (perf) can be success rate in (meta-)continualworld or
                # rewards in other environments
                perf, eps = agent.evaluate_cl(num_iterations=agent.config.evaluation_episodes)
                agent.task_eval_end()
                shell_eval_data[-1][eval_task_idx] = np.mean(perf)
                print(f'performance: {np.mean(perf)}')

            shell_eval_tracker = True
            shell_eval_end_time = time.time()
        
        ### EVALUATION BLOCK
        '''
        Performs the evaluation block.
        '''
        # If we want to stop evaluating then set agent.config.eval_interval to None
        '''if (agent.config.eval_interval is not None and shell_iterations % agent.config.eval_interval == 0):
            logger.info(Fore.BLUE + '*****agent {0} / evaluation block'.format(agent_id))
            _task_ids = shell_task_ids
            _tasks = shell_tasks
            _names = [eval_task_info['name'] for eval_task_info in _tasks]
            logger.info(Fore.BLUE + 'eval tasks: {0}'.format(', '.join(_names)))
            for eval_task_idx, eval_task_info in zip(_task_ids, _tasks):
                agent.task_eval_start(eval_task_info['task_label'])
                eval_states = agent.evaluation_env.reset_task(eval_task_info)
                agent.evaluation_states = eval_states
                # performance (perf) can be success rate in (meta-)continualworld or
                # rewards in other environments
                perf, eps = agent.evaluate_cl(num_iterations=agent.config.evaluation_episodes)
                agent.task_eval_end()
                shell_eval_data[-1][eval_task_idx] = np.mean(perf)
            shell_eval_tracker = True
            shell_eval_end_time = time.time()'''
        
        ###############################################################################
        # LOGGING
        if shell_eval_tracker:
            # log the last eval metrics to file
            _record = np.concatenate([shell_eval_data[-1], np.array(shell_eval_end_time).reshape(1,)])
            np.savetxt(eval_data_fh, _record.reshape(1, -1), delimiter=',', fmt='%.4f')
            del _record

            _metrics = shell_eval_data[-1]
            # compute icr
            icr = _metrics.sum()
            shell_metric_icr.append(icr)
            tb_writer.add_scalar('ShELL-ICR', icr, len(shell_eval_data))
            
            # log eval to file/screen and tensorboard
            logger.info('*****shell evaluation:')
            logger.info('shell eval ICR: {0}'.format(icr, shell_iterations))
            logger.info('shell eval TP: {0}'.format(np.sum(shell_metric_icr)))
            logger.scalar_summary('shell_eval/icr', icr)
            logger.scalar_summary('shell_eval/tpot', np.sum(shell_metric_icr))
            tb_writer.add_scalar('ShELL-TPOT', np.sum(shell_metric_icr), len(shell_eval_data))

            # reset eval tracker and add new buffer to save next eval metrics
            shell_eval_tracker = False
            shell_eval_data.append(np.zeros((num_eval_tasks, ), dtype=np.float32))


        #print('***** AGENT ITERATION TIME ELAPSED:', time.time()-START)

        # If ShELL is finished running all tasks then stop the program
        # this will have to be changed when we deploy so agents never stop working
        # and simply idle if there is nothing to learn.
        if shell_done:
            break
    # end of while True

    eval_data_fh.close()
    # discard last eval data entry as it was not used.
    if np.all(shell_eval_data[-1] == 0.):
        shell_eval_data.pop(-1)
    # save eval metrics
    to_save = np.stack(shell_eval_data, axis=0)
    with open(logger.log_dir + '/eval_metrics_agent_{0}.npy'.format(agent_id), 'wb') as f:
        np.save(f, to_save)

    agent.close()
    return


def trainer_learner_eps(agent, comm, agent_id, manager, mask_interval, mode):
    ###############################################################################
    ### Setup logger
    logger = agent.config.logger
    print(Fore.WHITE, end='') 
    logger.info('***** start l2d2-c training')


    ###############################################################################
    ### Setup trainer loop pre-requisites
    shell_done = False
    shell_episodes = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    #shell_task_ids = agent.config.task_ids
    shell_task_counter = 0
    #shell_eval_tracker = False
    shell_eval_data = []
    num_eval_tasks = len(agent.evaluation_env.get_all_tasks())
    shell_eval_data.append(np.zeros((num_eval_tasks, ), dtype=np.float32))
    #shell_metric_icr = [] # icr => instant cumulative reward metric. NOTE may be redundant now
    #eval_data_fh = open(logger.log_dir + '/eval_metrics_agent_{0}.csv'.format(agent_id), 'a', 
    #    buffering=1) # buffering=1 means flush data to file after every line written
    #shell_eval_end_time = None

    idling = True   # Flag to handle idling behaviour when curriculum ends.
    dict_to_query = None      # Variable to store the current task dictionary recrod to query for.


    ###############################################################################
    ### Select iteration logging function based on environment. Required for Meta World and Continual World
    if agent.__class__.__name__ == 'SACDetectShell':
        print('eps')
        itr_log_fn = _shell_eps_log
    
    elif agent.task.name == agent.config.ENV_METAWORLD or agent.task.name == agent.config.ENV_CONTINUALWORLD:
        itr_log_fn = _shell_itr_log_mw

    else:
        itr_log_fn = _shell_itr_log


    ###############################################################################
    ### Set the first task each agent is meant to train on
    states_ = agent.task.reset_task(shell_tasks[0])
    agent.states = agent.config.state_normalizer(states_)
    logger.info(f'***** ENVIRONMENT SWITCHING TASKS')
    logger.info(f'***** agent {agent_id} / setting first task (task 0)')
    logger.info(f"***** task: {shell_tasks[0]['task']}")
    logger.info(f"***** task_label: {shell_tasks[0]['task_label']}")

    # Set first task mask and record manually otherwise we run into issues with the implementation in the model.
    agent.task_train_start_emb(task_embedding=torch.zeros(agent.get_task_emb_size()))   # TODO: There is an issue with this which is that the first task will be set as zero and then the detect module with do some learning, find that the task does not match the zero embedding and start another task change. This leaves the first entry to a task change as useless. Also issues if we try to moving average this
    del states_


    ###############################################################################
    ### Start the comm module with the initial states and the first task label.
    # Returns shared queues to enable interaction between comm and agent.
    queue_label, queue_mask, queue_label_send, queue_mask_recv = comm.parallel(manager)


    ###############################################################################
    ### Logging setup (continued)
    tb_writer_emb = SummaryWriter(logger.log_dir + '/Detect_Component_Generated_Embeddings')
    _embeddings, _labels, exchanges, task_times, detect_module_activations = [], [], [], [], []
    task_times.append([0, shell_episodes, np.argmax(shell_tasks[0]['task_label'], axis=0), time.time()])


    ###############################################################################
    ### Comm module event handlers. These run in parallel to enable the interactions between the comm and agent.
    def mask_handler():
        """
        Handles incoming masks from other agents. Linearly combines masks and adds resulting mask to network.
        """
        while True:
            masks_list  = queue_mask.get()
            
            logger.info(Fore.WHITE + f'\n######### MASK RECEIVED FROM COMM #########')

            _masks = []
            _avg_embeddings = []
            _avg_rewards = []
            _mask_labels = []

            try:
                for mask_response_dict in masks_list:

                    mask = mask_response_dict['mask']
                    embedding = mask_response_dict['embedding']
                    reward = mask_response_dict['reward']
                    label = mask_response_dict['label']
                    ip = mask_response_dict['ip']
                    port = mask_response_dict['port']

                    _masks.append(mask)
                    _avg_embeddings.append(embedding)
                    _avg_rewards.append(reward)
                    _mask_labels.append(label)

                    # Log successful mask transfer
                    exchanges.append([shell_episodes, ip, port, np.argmax(label, axis=0), reward, len(mask), mask])
                    np.savetxt(logger.log_dir + '/exchanges_{0}.csv'.format(agent_id), exchanges, delimiter=',', fmt='%s')
            except Exception as e:
                traceback.print_exc()


            if len(_masks) > 0:
                try:
                    logger.info(Fore.WHITE + f'COMPOSING MASKS: {_avg_embeddings[0]}, {_avg_rewards[0]}, {_mask_labels[0]}')
                    logger.info(Fore.WHITE + f'MASK: {_masks[0]}')

                    # Update the knowledge base with the expected reward
                    agent.update_seen_tasks(_avg_embeddings[0], _avg_rewards[0], _mask_labels[0])#knowledge_base.update({tuple(label.tolist()): reward})
                    
                    # Update the network with the linearly combined mask
                    agent.consolidate_incoming(_masks)

                    logger.info(Fore.WHITE + 'COMPOSED MASK ADDED TO NETWORK!')
                except Exception as e:
                    traceback.print_exc()

    def conv_handler():
        """
        Handles interval label to mask conversions for outgoing mask responses.
        """
        while True:
            try:
                to_convert = queue_label_send.get()

                logger.info(Fore.WHITE + 'GOT ID TO CONVERT TO MASK')
                logger.info(f"MASK/TASK ID: {to_convert['sender_task_id']}")

                mask = agent.idx_to_mask(to_convert['sender_task_id'])
                reward = agent.seen_tasks['sender_task_id']['reward']
                emb = agent.seen_tasks['sender_task_id']['task_emb']
                label = agent.seen_tasks['sender_task_id']['ground_truth']
                print(Fore.LIGHTRED_EX + f'Found valid mask: {mask} with reward: {reward} and emb: {emb}')
                to_convert['response_mask'] = mask
                to_convert['response_reward'] = reward
                to_convert['response_embedding'] = emb
                to_convert['response_label'] = label
                queue_mask_recv.put((to_convert))
            except Exception as e:
                traceback.print_exc()


    ###############################################################################
    ### Start threads for the mask and conversion handlers.
    t_mask = mpd.Pool(processes=1)
    t_conv = mpd.Pool(processes=1)
    t_mask.apply_async(mask_handler)
    t_conv.apply_async(conv_handler)

    while True:
        start_time = time.time()
        ###############################################################################
        ### Idling behaviour. Idles until terminated when curriculum is completed.
        # While idling the agent acts as a server that can be queried for knowledge by
        # other agents.
        if shell_done:
            if idling:
                print('Agent is idling...') # Once idling the agent acts as a server that can be queried for knowledge until the agent encounters a new task (support not implemented yet)
                idling = False
                # Alternatively we can shutdown the agent here or do something for the experiment termination.
                
            #if omniscient_mode:
            #    if shell_iterations % mask_interval == 0:
            #        queue_label.put(None)

            #    shell_iterations += 1

            time.sleep(2) # Sleep value ensures the idling works as intended (fixes a problem encountered with the multiprocessing)
            continue
        print()


        ###############################################################################
        ### Iteration logging output.
        #logger.info(Fore.RED + 'GLOBAL REGISTRY (seen_tasks dict)')
        #for key, val in agent.seen_tasks.items():
        #    logger.info(f"{key} --> embedding: {val['task_emb']}, reward: {val['reward']}, ground truth task id: {np.argmax(val['ground_truth'], axis=0)}")


        ###############################################################################
        ### Query for knowledge using communication process. Send label/embedding to the communication module to query for relevant knowledge from other peers.
        if dict_to_query is not None:
            if shell_episodes % mask_interval == 0:
                # Approach 2: At this point consolidate masks and then we can reset beta parameters. Then we can get new masks from network and combine.
                
                queue_label.put(dict_to_query)


        ###############################################################################
        ### Agent training iteration: collect on policy data and optimise the agent
        dict_logs = agent.episode()
        shell_episodes += 1

        
        ###############################################################################
        ### Run detect module. Generates embedding for SAR. Perform check to see if there has been a task change or not.
        '''_dist_threshold = agent.emb_dist_threshold
        if shell_episodes != 0 and shell_episodes % agent.detect_module_activation_frequency == 0 and agent.data_buffer.size() >= (agent.detect.get_num_samples()):
            # Run the detect module on SAR and return some logging output.
            task_change_flag, new_emb, ground_truth_task_label, emb_dist, emb_bool, agent_seen_tasks = run_detect_module(agent)
            
            if task_change_flag:
                logger.info(Fore.YELLOW + f'TASK CHANGE DETECTED! NEW MASK CREATED. CURRENT TASK INDEX: {agent.current_task_key}')
            
            # Update the dictionary containing the current task embedding to query for.
            dict_to_query = agent.seen_tasks[agent.current_task_key]

            # Log the operation of the detect module. Currently this is broken (requires a custom tensorboard solution. ask Chris)
            log_string = f'Time: {time.time()}, Iteration: {shell_episodes}, Detect activation: {True}, Num samples for detection: {agent.detect.get_num_samples()}, Task change flag: {task_change_flag}, New embedding: {new_emb}, Ground truth label: {ground_truth_task_label}, Current embedding: {agent.current_task_emb}, Threshold: {_dist_threshold}, Distance: {emb_dist}, Embedding similarity: {emb_bool}, Agent seen tasks: {agent_seen_tasks}'
            detect_module_activations.append([log_string])
            np.savetxt(logger.log_dir + '/detect_activations_{0}.csv'.format(agent_id), detect_module_activations, delimiter=',', fmt='%s')
            
            # Logging embeddings and labels
            if new_emb is not None:
                _label = torch.tensor(np.array([ground_truth_task_label]))
                _embeddings.append(new_emb)
                _labels.append(_label)
                emb_t = torch.stack(tuple(_embeddings))
                #l_t = torch.stack(tuple(_labels))
                logger.info(Fore.WHITE + f'Embedding: {new_emb}\nLabel: {_label}\nDistance: {emb_dist}, Threshold: {agent.emb_dist_threshold}\n')
                tb_writer_emb.add_embedding(emb_t, metadata=_labels, global_step=shell_episodes)'''


        ###############################################################################
        ### Logs metrics to tensorboard log file and updates the embedding, reward pair in this cycle for a particular task.
        if shell_episodes % agent.config.iteration_log_interval == 0:
            #print(dict_logs)
            itr_log_fn(logger, agent, agent_id, shell_episodes, shell_task_counter, dict_logs)
                
            # Save agent model
            agent.save(agent.config.log_dir + '/%s-%s-model-%s.bin' % (agent.config.agent_name, agent.config.tag, agent.task.name))


        ###############################################################################
        ### Environment task change at the end of the max steps for each task. Agent is not aware of this change and must detect it using the detect module.
        if not agent.config.max_steps: raise ValueError('`max_steps` should be set for each agent')
        task_steps_limit = agent.config.max_steps[shell_task_counter] * (shell_task_counter + 1)

        # If agent completes the maximum number of steps for a task then switch to the next task in the curriculum.
        if agent.total_steps >= task_steps_limit:
            task_counter_ = shell_task_counter
            logger.info('\n' + Fore.WHITE + f'*****agent {agent_id} / end of training on task {task_counter_}')
            
            #MOVED to Assing EMB in PPO agent.task_train_end()

            # Increment task counter
            task_counter_ += 1
            shell_task_counter = task_counter_

            # If curriculum is not completed, switch to the next task in the curriculum
            if task_counter_ < len(shell_tasks):
                # new task
                logger.info(Fore.WHITE + f'***** ENVIRONMENT SWITCHING TASKS')
                logger.info(Fore.WHITE + f'***** agent {agent_id} / set next task {task_counter_}')
                logger.info(Fore.WHITE + f"***** task: {shell_tasks[task_counter_]['task']}")
                logger.info(Fore.WHITE + f"***** task_label: {shell_tasks[task_counter_]['task_label']}")
                
                # Set the new task from the environment. Agent remains unaware of this change and will continue until
                # detect module detects distrubtion shift.
                states_ = agent.task.reset_task(shell_tasks[task_counter_]) # reset_task sets the new task and returns the reset intial states.
                agent.states = agent.config.state_normalizer(states_)
                
                #MOVED to Assing EMB in PPO agent agent.task_train_start(shell_tasks[task_counter_]['task_label'])

                del states_

                task_times.append([task_counter_, shell_episodes, np.argmax(shell_tasks[task_counter_]['task_label'], axis=0), time.time()])
                np.savetxt(logger.log_dir + '/task_changes_{0}.csv'.format(agent_id), task_times, delimiter=',', fmt='%s')

            else:
                shell_done = True # training done for all task for agent. This leads to the idling behaviour in next iteration.
                logger.info(f'*****agent {agent_id} / end of all training')

            del task_counter_


        logger.info(f'----------------------- Episode complete in {time.time() - start_time} seconds -----------------------\n')





#########################################################################################################################################
########################################  U T I L I T Y      F U N C T I O N S  #########################################################
#########################################################################################################################################


def detect_module_activation_check(shell_training_iterations, detect_module_activation_frequncy, agent):
    '''
    Utlity function to check whether the Detect module should be activated or not based on conditions:
        shell_training_iterations
        detect activation frequency
        data buffer size

    It makes sure that the data buffer has the necessary amount of data for the detect module to sample from
    '''
    

    #print("REPAY_BUFFER_SIZE:", agent.data_buffer.size())
    #print("DETECT_MODULE_SAMPLE_SIZE:", agent.detect.get_num_samples())
    #print("ACTUAL_SAMPLES_DETECT_NEEDS:", (agent.detect.get_num_samples() * 64))
    if shell_training_iterations != 0 and shell_training_iterations % detect_module_activation_frequncy == 0 and agent.data_buffer.size() >= (agent.detect.get_num_samples() ):#Times the batch size the Detect Module uses.
        return True
    else:
        return False
        

def run_detect_module(agent):
    '''Uitility function for running all the necassery methods and function for the detect module
    so the approprate embeddings are generated for each batch of SAR data'''
    
    #Initilize the retun varibles with None values in the case of the detect module not being appropriate to run.
    task_change_detected, emb_dist, emb_bool, ground_truth_task_label = None, None, None, torch.tensor(0)
    emb_dist, m_dist1, m_dist2, cos_sim, density, emd, wasserstein_distance = 0, 0, 0, 0, 0, 0, 0
    
    # Extract SAR data from agent's replay buffer
    sar_data = agent.extract_sar()

    #if len(agent.embedding_history) == 0:
    #    agent.embedding_history.append(agent.current_task_emb)

    # Produce task embedding from sar data using 
    new_embedding = agent.compute_task_embedding(sar_data, agent.get_task_action_space_size())
    #new_embedding = torch.from_numpy(agent.detect.online_pca(new_embedding))
    #new_embedding = agent.detect.random_projections(new_embedding, 512)
    #new_embedding = agent.detect.encode_embedding(new_embedding, 128)
    #print(new_embedding)
    #EAVG = sum(agent.embedding_history) / len(agent.embedding_history)
    #emb_dist = agent.detect.emb_distance(EAVG, new_embedding)
    #agent.embedding_history.append(new_embedding)

    
    # Add up history of embeddings to rewards over time.
    #agent.detect.add_embedding(new_embedding, np.mean(agent.iteration_rewards))

    current_embedding = agent.current_task_emb
    ground_truth_task_label = agent.get_current_task_label()

    # Compute distance
    emb_bool = current_embedding == new_embedding
    print(f'IN RUN DETECT MODULE CURRENT: {current_embedding},  NEW: {new_embedding}')
    emb_dist = agent.detect.emb_distance(current_embedding, new_embedding)

    # Mahalanobis distance from identity matrix
    #cov_matrix = np.eye(len(new_embedding))
    #m_dist1 = agent.detect.mahalanobis_distance(current_embedding, new_embedding, cov_matrix)

    # Mahalanobis distance from mean covariance matrix
    #embeddings = np.vstack([new_embedding, current_embedding])
    #mean_vector = np.mean(embeddings, axis=0)
    #centered_embeddings = embeddings - mean_vector
    #cov_matrix = np.cov(centered_embeddings, rowvar=False)
    #m_dist2 = agent.detect.mahalanobis_distance(current_embedding, new_embedding, cov_matrix)

    # Cosine similarity
    #cos_sim = agent.detect.cosine_sim(current_embedding, new_embedding)

    # Kernel Density Estimation
    #density = agent.detect.estimate_density(current_embedding, new_embedding, 0.5)

    # Wasserstein distance / Earth Mover's Distance
    #emd = agent.detect.wasserstein_distance(current_embedding, new_embedding)




    task_change_detected = agent.assign_task_emb(new_embedding, emb_dist)      # Task change detection purely on thresholding
    #task_change_detected = agent.assign_task_emb_birch(new_embedding)          # Task change detection using BIRCH clustering TODO: Needs fixing
    #task_change_detected = agent.assign_task_weighted_avg(new_embedding)       # Task change detection using a normalized reward weighted distance history
    #task_change_detected = agent.assign_task_emb_cusum(new_embedding, emb_dist) # Task change detection using CUSUM algorithm with a last 10 timestep distance history

    #if task_change_detected:
    #    agent.embedding_history = [torch.zeros(agent.get_task_emb_size()), agent.embedding_history[-1]]

    # Get the updated task embedding from agent
    agent_seen_tasks = agent.get_seen_tasks()

    distances = [emb_dist, m_dist1, m_dist2, cos_sim, density, emd, wasserstein_distance]

    return task_change_detected, new_embedding, ground_truth_task_label, distances, emb_bool, agent_seen_tasks