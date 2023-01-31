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
import pickle
import os
import time
import datetime
import torch
from .torch_utils import *
from torch.utils.tensorboard import SummaryWriter
from ..shell_modules import *

import multiprocessing.dummy as mpd
from queue import Empty
from colorama import Fore
import traceback

try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path


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

    for key, value in dict_logs.items():
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
            req['mask'] = agent.label_to_mask(req['task_label'])
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
def shell_dist_train_mp(agent, comm, agent_id, num_agents, manager, knowledge_base, mask_interval, omniscient_mode):
    logger = agent.config.logger

    logger.info(Fore.BLUE + '*****start shell training')

    shell_done = False
    shell_iterations = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    shell_task_ids = agent.config.task_ids
    shell_task_counter = 0

    shell_agent_seed = agent.config.seed        # Chris


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

    await_response = [True,] * num_agents # flag
    # set the first task each agent is meant to train on
    states_ = agent.task.reset_task(shell_tasks[0])
    agent.states = agent.config.state_normalizer(states_)
    logger.info(Fore.BLUE + '*****agent {0} / setting first task (task 0)'.format(agent_id))
    logger.info(Fore.BLUE + 'task: {0}'.format(shell_tasks[0]['task']))
    logger.info(Fore.BLUE + 'task_label: {0}'.format(shell_tasks[0]['task_label']))
    agent.task_train_start(shell_tasks[0]['task_label'])
    #print()
    del states_


    tb_writer_emb = SummaryWriter(logger.log_dir + '/Detect_Component_Generated_Embeddings') #Chris
    ql = []
    ll =[]

    # Msg can be embedding or task label.
    # Set msg to first task. The agents will then request knowledge on the first task.
    # this will ensure that other agents are aware that this agent is now working this task
    # until a task change happens.
    msg = None#shell_tasks[0]['task_label']

    # Track which agents are working which tasks. This will be resized every time a new agent is added
    # to the network. Every time there is a communication step 1, we will check if it is none otherwise update
    # this dictionary
    #track_tasks = {agent_id: torch.from_numpy(msg)}


    # Agent-Communication interaction queues
    queue_mask = manager.Queue()
    queue_label = manager.Queue()
    queue_label_send = manager.Queue()  # Used to send label from comm to agent to convert to mask
    queue_mask_recv = manager.Queue()   # Used to send mask from agent to comm after conversion from label

    # Put in the initial data into the loop queue so that the comm module is not blocking the agent
    # from running.
    #queue_loop.put_nowait((shell_iterations))

    # Start the communication module with the initial states and the first task label.
    # Get the mask ahead of the start of the agent iteration loop so that it is available sooner
    # Also pass the queue proxies to enable interaction between the communication module and the agent module
    comm.parallel(queue_label, queue_mask, queue_label_send, queue_mask_recv)

    exchanges = []
    task_times = []
    detect_module_activations = [] #Chris
    task_times.append([0, shell_iterations, np.argmax(shell_tasks[0]['task_label'], axis=0), time.time()])


    # Communication module interaction handlers
    def mask_handler():
        """
        Handles incoming masks from other agents. Distills the mask knowledge to the agent's network.
        """
        while True:
            mask, label, reward, ip, port  = queue_mask.get()
            logger.info(Fore.WHITE + f'\nReceived mask: \nMask:{mask}\nReward:{reward}\nSrc:{ip, port}\nEmbedding:{label}')
                
            if mask is not None:
                # Update the knowledge base with the expected reward
                knowledge_base.update({tuple(label.tolist()): reward})
                # Update the network with the mask
                agent.distil_task_knowledge_single(mask, label)

                logger.info(Fore.WHITE + 'KNOWLEDGE DISTILLED TO NETWORK!')

                # Mask transfer logging.
                exchanges.append([shell_iterations, ip, port, np.argmax(label, axis=0), reward, len(mask), mask])
                np.savetxt(logger.log_dir + '/exchanges_{0}.csv'.format(agent_id), exchanges, delimiter=',', fmt='%s')

    def conv_handler():
        """
        Handles interval label to mask conversions for outgoing mask responses.
        """
        while True:
            convert = queue_label_send.get()
            logger.info('Got label to convert to mask')
            print(convert['embedding'], type(convert['embedding']))

            convert['mask'] = agent.label_to_mask(convert['embedding'].detach().cpu().numpy())
            queue_mask_recv.put((convert))

    # Start threads for the mask and conversion handlers.
    t_mask = mpd.Pool(processes=1)
    t_conv = mpd.Pool(processes=1)
    t_mask.apply_async(mask_handler)
    t_conv.apply_async(conv_handler)

    idling = True

    while True:
        if shell_done:
            if idling:
                print('Agent is idling...') # Useful visualisation but we can eventually remove this idling message.
                idling = False
            if omniscient_mode:
                if shell_iterations % mask_interval == 0:
                    queue_label.put(None)

                shell_iterations += 1

            time.sleep(2)
            continue

        #print(f'Knowledge base in agent: {knowledge_base}')
        logger.info(f'{Fore.RED}World: {comm.world_size.value}')
        #logger.info(f'{Fore.RED}Meta: {len(comm.metadata)}')
        #for key, val in comm.knowledge_base.items(): print(f'{Fore.RED}Knowledge base: {key}: {val}')
        for addr in comm.query_list: print(f'{Fore.RED}{addr[0]}, {addr[1]}')

        # Send label/embedding to the communication module to query for relevant knowledge from other peers.
        if msg is not None:
            if shell_iterations % mask_interval == 0:
                queue_label.put(msg)

        
        ### AGENT ITERATION (training step): collect on policy data and optimise the agent
        '''
        Handles the data collection and optimisation within the agent.
        TODO: Look into multihreading/multiprocessing the data collection and optimisation of the agent
                to achieve the continous data collection we want for a real world scenario.
            
            Possibly the optimisation could be made a seperate process parallel to the data collection
                process, similar to the communication-agent(trainer) architecture. Data collection and the code
                below would run together in the main loop.
        '''
        dict_logs = agent.iteration()
        shell_iterations += 1

        


        activation_flag = detect_module_activation_check(shell_iterations, agent.get_detect_module_activation_frequency(), agent)   #Chris
        str_task_chng_msg, task_change_flag, emb_dist, emb_bool, new_emb, current_task_embedding, ground_truth_task_label= run_detect_module(agent, activation_flag)   #Chris
        msg = current_task_embedding

        #Logging of the detect operations!!!   #Chris
        if activation_flag:
            print(Fore.GREEN)
            detect_module_activations.append([shell_iterations, activation_flag, str_task_chng_msg, task_change_flag, "Number of Samples for detection:", agent.detect.get_num_samples(), "I AM THE NEW EMBEDING!:", new_emb, 'Hi I am the GroundTruth LABEL:', ground_truth_task_label, "Hi I AM THE CURRENT TASK EMBEDDING!", current_task_embedding, "HI I AM DIST:", emb_dist, "HI I CHECK EQ CURR VS NEW EMB:", emb_bool])
            np.savetxt(logger.log_dir + '/detect_activations_{0}.csv'.format(agent_id), detect_module_activations, delimiter=',', fmt='%s')
            if new_emb is not None:
                q = new_emb#torch.Tensor.unsqueeze(new_emb, 0)
                l = torch.tensor([ground_truth_task_label])
                ql.append(q)
                ll.append(l)
                qlt = tuple(ql)
                llt = tuple(ll)
                emb_t = torch.stack(qlt)
                l_t = torch.stack(llt)
                print("EMB_T:", emb_t)
                print(f"L_T: {l_t}")
                print("HELLO FROM THE DARK SIDE... {}, I MUST HAVE CALL A THOUSAND TIMES: {}".format(q, l))
                tb_writer_emb.add_embedding(emb_t, metadata=ll)
                #tb_writer_emb.close()



        ### TENSORBOARD LOGGING & SELF TASK REWARD TRACKING
        '''
        Logs metrics to tensorboard log file and updates the embedding, reward pair in this cycle for a particular task.
        '''
        if shell_iterations % agent.config.iteration_log_interval == 0:
            itr_log_fn(logger, agent, agent_id, shell_iterations, shell_task_counter, dict_logs)
            
            
            if msg is not None:
                # Create a dictionary to store the most recent iteration rewards for a mask. Update in every iteration
                # logging cycle. Take average of all worker averages as the most recent reward score for a given task
                knowledge_base[tuple(msg.tolist())] = np.around(np.mean(agent.iteration_rewards), decimals=6)
                logger.info(f'{msg} is logged to knowledge base')
                
            # Save agent model
            agent.save(agent.config.log_dir + '/%s-%s-model-%s.bin' % (agent.config.agent_name, agent.config.tag, \
                    agent.task.name))


        ### EVALUATION BLOCK    Deprecated.
        '''
        Performs the evaluation block. Has been replaced by the evaluation agent. Can be re-enabled by changing the eval interval value in the configuration in run_shell_dist_mp.py.
        '''
        # If we want to stop evaluating then set agent.config.eval_interval to None
        if (agent.config.eval_interval is not None and shell_iterations % agent.config.eval_interval == 0):
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


        ### TASK CHANGE
        # end of current task training. move onto next task or end training if last task.
        # i.e., Task Change occurs here. For detect module, if the task embedding signifies a task
        # change then that should occur here.
        '''
        If we want to use a Fetch All mode for ShELL then we need to add a commmunication component
        at task change which broadcasts the mask to all other agents currently on the network.

        Otherwise the current implementation is a On Demand mode where each agent requests knowledge
        only when required.
        '''
        if not agent.config.max_steps: raise ValueError('`max_steps` should be set for each agent')
        task_steps_limit = agent.config.max_steps[shell_task_counter] * (shell_task_counter + 1)
        if agent.total_steps >= task_steps_limit:
            task_counter_ = shell_task_counter
            logger.info('\n' + Fore.BLUE + '*****agent {0} / end of training on task {1}'.format(agent_id, task_counter_))
            #MOVED to Assing EMB in PPO agent.task_train_end()

            task_counter_ += 1
            shell_task_counter = task_counter_
            if task_counter_ < len(shell_tasks):
                task_times.append([task_counter_, shell_iterations, np.argmax(shell_tasks[task_counter_]['task_label'], axis=0), time.time()])
                np.savetxt(logger.log_dir + '/task_changes_{0}.csv'.format(agent_id), task_times, delimiter=',', fmt='%s')


                # new task
                logger.info(Fore.BLUE + '*****agent {0} / set next task {1}'.format(agent_id, task_counter_))
                logger.info(Fore.BLUE + 'task: {0}'.format(shell_tasks[task_counter_]['task']))
                logger.info(Fore.BLUE + 'task_label: {0}'.format(shell_tasks[task_counter_]['task_label']))
                states_ = agent.task.reset_task(shell_tasks[task_counter_]) # set new task
                agent.states = agent.config.state_normalizer(states_)
                #MOVED to Assing EMB in PPO agent agent.task_train_start(shell_tasks[task_counter_]['task_label'])

                # set message (task_label) that will be sent to other agent as a request for
                # task knowledge (mask) about current task. this will be sent in the next
                # receive/send request cycle.
                logger.info(Fore.BLUE + '*****agent {0} / query other agents using current task label'\
                    .format(agent_id))

                # Update the msg, track_tasks dict for this agent and reset await_responses for new task
                #msg = shell_tasks[task_counter_]['task_label']
                #track_tasks[agent_id] = torch.from_numpy(msg)       # remove later
                await_response = [True,] * num_agents
                del states_
                #print()
            else:
                shell_done = True # training done for all task for agent
                logger.info(Fore.BLUE + '*****agent {0} / end of all training'.format(agent_id))
            del task_counter_


        ### EVALUATION BLOCK LOGGING    Deprecated.
        '''
        Logs the evaluation block data to a text file. No longer necessary as we are now using the evaluation agent.
        '''
        if shell_eval_tracker:
            # log the last eval metrics to file
            _record = np.concatenate([shell_eval_data[-1],np.array(shell_eval_end_time).reshape(1,)])
            np.savetxt(eval_data_fh, _record.reshape(1, -1), delimiter=',', fmt='%.4f')
            del _record

            # reset eval tracker and add new buffer to save next eval metrics
            shell_eval_tracker = False
            shell_eval_data.append(np.zeros((num_eval_tasks, ), dtype=np.float32))

'''
shell evaluation: concurrent processing for event-based communication.
'''
def shell_dist_eval_mp(agent, comm, agent_id, num_agents, manager, knowledge_base):

    #Create a SmumaryWriter object for the Evaluation Agent

    

    logger = agent.config.logger
    #print()

    logger.info(Fore.BLUE + '*****start shell training')

    tb_writer = SummaryWriter(logger.log_dir + '/DMIU-ShELL_metrics')

    shell_done = False
    shell_iterations = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    shell_task_ids = agent.config.task_ids
    shell_task_counter = 0


    shell_agent_seed = agent.config.seed        # Chris
    agent.config.eval_interval = 1      # Manual overide
    


    _task_ids = shell_task_ids
    _tasks = shell_tasks
    # agent.seen_tasks = {_tasks}
    agent.seen_tasks = {}
    for idx, eval_task_info in enumerate(_tasks):
        agent.seen_tasks[idx] =  eval_task_info['task_label']
    _names = [eval_task_info['name'] for eval_task_info in _tasks]
    eval_task_info = zip(_task_ids, _tasks)

    shell_eval_tracker = False
    shell_eval_data = []
    num_eval_tasks = len(agent.evaluation_env.get_all_tasks())
    shell_eval_data.append(np.zeros((num_eval_tasks, ), dtype=np.float32))
    shell_metric_icr = [] # icr => instant cumulative reward metric. NOTE may be redundant now
    eval_data_fh = open(logger.log_dir + '/eval_metrics_agent_{0}.csv'.format(agent_id), 'a', \
        buffering=1) # buffering=1 means flush data to file after every line written
    shell_eval_end_time = None

    # Evaluation agent likely wont need a iteration logging function
    '''if agent.task.name == agent.config.ENV_METAWORLD or \
        agent.task.name == agent.config.ENV_CONTINUALWORLD:
        itr_log_fn = _shell_itr_log_mw
    else:
        itr_log_fn = _shell_itr_log'''
    
    task_steps_limit = sum(agent.config.max_steps)


    # Msg can be embedding or task label.
    # Set msg to first task. The agents will then request knowledge on the first task.
    # this will ensure that other agents are aware that this agent is now working this task
    # until a task change happens.
    msg = _tasks[0]['task_label']

    queue_mask = manager.Queue()
    queue_label = manager.Queue()
    queue_label_send = manager.Queue()  # Used to send label from comm to agent to convert to mask
    queue_mask_recv = manager.Queue()   # Used to send mask from agent to comm after conversion from label

    # Start the communication module with the initial states and the first task label.
    # Get the mask ahead of the start of the agent iteration loop so that it is available sooner
    # Also pass the queue proxies to enable interaction between the communication module and the agent module
    comm.parallel(queue_label, queue_mask, queue_label_send, queue_mask_recv)

    exchanges = []
    task_times = []
    detect_module_activations = [] #Chris
    task_times.append([0, shell_iterations, np.argmax(shell_tasks[0]['task_label'], axis=0), time.time()])


    # Communication module interaction handlers
    def mask_handler():
        """
        Handles incoming masks from other agents. Distills the mask knowledge to the agent's network.
        """
        while True:
            mask, label, reward, ip, port  = queue_mask.get()
            logger.info(Fore.WHITE + f'\nReceived mask: \nMask:{mask}\nReward:{reward}\nSrc:{ip, port}\nEmbedding:{label}')
                
            if mask is not None:
                # Update the knowledge base with the expected reward
                knowledge_base.update({label: reward})
                # Update the network with the mask
                agent.distil_task_knowledge_single(mask, label)

                logger.info(Fore.WHITE + 'KNOWLEDGE DISTILLED TO NETWORK!')

                # Mask transfer logging.
                exchanges.append([shell_iterations, ip, port, np.argmax(label, axis=0), reward, len(mask), mask])
                np.savetxt(logger.log_dir + '/exchanges_{0}.csv'.format(agent_id), exchanges, delimiter=',', fmt='%s')

    t_mask = mpd.Pool(processes=1)
    t_mask.apply_async(mask_handler)


    while True:
        print()
        print(shell_iterations, shell_task_counter, agent.total_steps, agent.config.max_steps, task_steps_limit)
        agent.total_steps += agent.config.rollout_length * agent.config.num_workers
        
        # Send the msg of this iteration. It will be either a task label or NoneType. Eitherway
        # the communication module will do its thing.
        msg = _tasks[shell_task_counter]['task_label']
        #agent.curr_eval_task_label = msg
        queue_label.put(msg)


        # Evaluation agent does not need to do data collection and optimisation so we will
        # not need to run the agent.iteration() function here. We will need to increment the shell
        # iterations however.
        #dict_logs = agent.iteration()
        #perf, eps = agent.evaluate_cl(num_iterations=agent.config.evaluation_episodes)
        shell_iterations += 1


        
        ### EVALUATION BLOCK
        '''
        Performs the evaluation block.
        '''
        # If we want to stop evaluating then set agent.config.eval_interval to None
        if (agent.config.eval_interval is not None and shell_iterations % agent.config.eval_interval == 0):
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


        task_counter_ = shell_task_counter
        task_counter_ += 1
        if task_counter_ > len(shell_tasks)-1:
            task_counter_ = 0

        shell_task_counter = task_counter_


        ### EVALUATION BLOCK LOGGING
        '''
        Logs the evaluation block data and tracks it.
        '''
        if shell_eval_tracker:
            # log the last eval metrics to file
            _record = np.concatenate([shell_eval_data[-1],np.array(shell_eval_end_time).reshape(1,)])
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

#########################################################################################################################################
########################################  U T I L I T Y      F U N C T I O N S  #########################################################
#########################################################################################################################################


def detect_module_activation_check(shell_training_iterations, detect_module_activation_frequncy, agent):
    '''Utility function for checking wheter the Detect Module should be activated or not based on the
    shell_training_iterations of the agent and the activation frequncy given from the configuration file.
    It makes sure that the data buffer has the ammount of data necassary for the detect module to sample.'''
    

    print("REPAY_BUFFER_SIZE:", agent.data_buffer.size())
    print("DETECT_MODULE_SAMPLE_SIZE:", agent.detect.get_num_samples())
    #print("ACTUAL_SAMPLES_DETECT_NEEDS:", (agent.detect.get_num_samples() * 64))
    if shell_training_iterations != 0 and shell_training_iterations % detect_module_activation_frequncy == 0 and agent.data_buffer.size() >= (agent.detect.get_num_samples() ):#Times the batch size the Detect Module uses.
        return True
    else:
        return False
        

def run_detect_module(an_agent, activation_check_flag):
    '''Uitility function for running all the necassery methods and function for the detect module
    so the approprate embeddings are generated for each batch of SAR data'''
    
    #Initilize the retun varibles with None values in the case of the detect module not being appropriate to run.
    str_task_chng_msg, task_change_flag, emb_dist, emb_bool, new_emb, current_task_embedding, ground_truth_task_label = None, None, None, None, None, None, torch.tensor(0)
    
    if activation_check_flag:
        sar_data = an_agent.sar_data_extraction()
        print("SAR_DATA_ARR:", sar_data)
        print("SAR SIZE:",sar_data.shape)
        print(sar_data)
        #print("CURR_EMB:", current_embedding)
        new_emb = an_agent.compute_task_embedding(sar_data, an_agent.get_task_action_space_size())
        current_embedding = an_agent.get_current_task_embedding()
        print("CURR_EMB:", current_embedding)
        print("NEW_EMB:", new_emb)
        print("EMBEDDING SIZE:", len(new_emb))
        ground_truth_task_label = an_agent.get_current_task_label()
        print("Ground_Truth_Task_Label:", ground_truth_task_label)
        print(type(ground_truth_task_label))
        emb_bool = current_embedding == new_emb
        emb_dist = an_agent.calculate_emb_distance(current_embedding, new_emb)
        emb_dist_thrshld = an_agent.get_emb_dist_threshold()
        str_task_chng_msg, task_change_flag = an_agent.assign_task_emb(new_emb, emb_dist, emb_dist_thrshld)
        current_task_embedding = an_agent.get_current_task_embedding()
        an_agent_seen_tasks = an_agent.get_seen_tasks()
        print("/n/n/nAgent SEEN TASKS:", an_agent_seen_tasks, "/n/n/n/n")

    return str_task_chng_msg, task_change_flag, emb_dist, emb_bool, new_emb, current_task_embedding, ground_truth_task_label#str_task_chng_msg, task_change_flag, emb_dist, emb_bool, new_emb    