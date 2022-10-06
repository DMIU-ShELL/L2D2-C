#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


'''

__________.__                           __  .__    .__                                       
\______   \  |   ____   ______ ______ _/  |_|  |__ |__| ______   _____   ____   ______ ______
 |    |  _/  | _/ __ \ /  ___//  ___/ \   __\  |  \|  |/  ___/  /     \_/ __ \ /  ___//  ___/
 |    |   \  |_\  ___/ \___ \ \___ \   |  | |   Y  \  |\___ \  |  Y Y  \  ___/ \___ \ \___ \ 
 |______  /____/\___  >____  >____  >  |__| |___|  /__/____  > |__|_|  /\___  >____  >____  >
        \/          \/     \/     \/             \/        \/        \/     \/     \/     \/

                        and may god help you if you try to understand it.
'''

import numpy as np
import pickle
import os
import time
import datetime
import torch
from .torch_utils import *
from ..shell_modules import *

try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path


def _shell_itr_log(logger, agent, agent_idx, itr_counter, task_counter, dict_logs):
    logger.info('agent %d, task %d / iteration %d, total steps %d, ' \
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
    logger.info('agent %d, task %d / iteration %d, total steps %d, ' \
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

'''
shell training: single processing, all agents are executed/trained in a single process
with each agent taking turns to execute a training step.

note: task boundaries are explictly given to the agents. this means that each agent
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

    if agents[0].task.name == agent[0].config.ENV_METAWORLD or \
        agent[0].task.name == agent[0].config.ENV_CONTINUALWORLD:
        itr_log_fn = _shell_itr_log_mw
    else:
        itr_log_fn = _shell_itr_log

    print()
    logger.info('*****start shell training')

    # set the first task each agent is meant to train on
    for agent_idx, agent in enumerate(agents):
        states_ = agent.task.reset_task(shell_tasks[agent_idx][0])
        agent.states = agent.config.state_normalizer(states_)
        logger.info('*****agent {0} /setting first task (task 0)'.format(agent_idx))
        logger.info('task: {0}'.format(shell_tasks[agent_idx][0]['task']))
        logger.info('task_label: {0}'.format(shell_tasks[agent_idx][0]['task_label']))
        agent.task_train_start(shell_tasks[agent_idx][0]['task_label'])
        print()
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
                logger.info('*****agent {0} / evaluation block'.format(agent_idx))
                _task_ids = shell_task_ids[agent_idx]
                _tasks = shell_tasks[agent_idx]
                _names = [eval_task_info['name'] for eval_task_info in _tasks]
                logger.info('eval tasks: {0}'.format(', '.join(_names)))
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
                raise ValueError('`max_steps` should be set for each agent')
            task_steps_limit = agent.config.max_steps[shell_task_counter[agent_idx]] * \
                (shell_task_counter[agent_idx] + 1)
            if agent.total_steps >= task_steps_limit:
                print()
                task_counter_ = shell_task_counter[agent_idx]
                logger.info('*****agent {0} / end of training on task {1}'.format(agent_idx, \
                    task_counter_))
                agent.task_train_end()

                task_counter_ += 1
                shell_task_counter[agent_idx] = task_counter_
                if task_counter_ < len(shell_tasks[agent_idx]):
                    logger.info('*****agent {0} / set next task {1}'.format(agent_idx, task_counter_))
                    logger.info('task: {0}'.format(shell_tasks[agent_idx][task_counter_]['task']))
                    logger.info('task_label: {0}'.format(shell_tasks[agent_idx][task_counter_]['task_label']))
                    states_ = agent.task.reset_task(shell_tasks[agent_idx][task_counter_]) # set new task
                    agent.states = agent.config.state_normalizer(states_)
                    agent.task_train_start(shell_tasks[agent_idx][task_counter_]['task_label'])

                    # ping other agents to see if they have knoweledge (mask) of current task
                    logger.info('*****agent {0} / pinging other agents to leverage on existing ' \
                        'knowledge about task'.format(agent_idx))
                    a_idxs = list(range(len(agents)))
                    a_idxs.remove(agent_idx)
                    other_agents = [agents[i] for i in a_idxs]
                    found_knowledge = agent.ping_agents(other_agents)
                    if found_knowledge > 0:
                        logger.info('found knowledge about task from other agents')
                        logger.info('number of task knowledge found: {0}'.format(found_knowledge))
                    else:
                        logger.info('could not find any agent with knowledge about task')
                    del states_
                    print()
                else:
                    shell_done[agent_idx] = True # training done for all task for agent
                    logger.info('*****agent {0} / end of all training'.format(agent_idx))
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
            logger.info('*****shell evaluation:')
            logger.info('best agent per task:'.format(_agent_ids))
            logger.info('shell eval ICR: {0}'.format(icr))
            logger.info('shell eval TP: {0}'.format(np.sum(shell_metric_icr)))
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

note: task boundaries are explictly given to the agents. this means that each agent
knows when task changes. 
'''
def shell_dist_train(agent, comm, agent_id, num_agents):
    logger = agent.config.logger
    print()
    logger.info('*****start shell training')

    shell_done = False
    shell_iterations = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    shell_task_ids = agent.config.task_ids
    shell_task_counter = 0


    mask_rewards_dict = dict()


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
    logger.info('*****agent {0} / setting first task (task 0)'.format(agent_id))
    logger.info('task: {0}'.format(shell_tasks[0]['task']))
    logger.info('task_label: {0}'.format(shell_tasks[0]['task_label']))
    agent.task_train_start(shell_tasks[0]['task_label'])
    print()
    del states_

    msg = None
    while True:
        # listen for and send info (task label or NULL message) to other agents
        other_agents_request = comm.send_receive_request(msg)
        print(other_agents_request)
        msg = None # reset message

        requests = []
        for req in other_agents_request:
            print(req)
            if req is None: continue
            req['mask'] = agent.label_to_mask(req['task_label'])
            requests.append(req)
        if len(requests) > 0:
            comm.send_response(requests)

        # if the agent earlier sent a request, check whether response has been sent.
        if any(await_response):
            logger.info('awaiting response: {0}'.format(await_response))
            masks = []
            received_masks = comm.receive_response()
            for i in range(len(await_response)):
                if await_response[i] is False: continue

                if received_masks[i] is False: continue
                elif received_masks[i] is None: await_response[i] = False
                else:
                    masks.append(received_masks[i])
                    await_response[i] = False
            logger.info('number of task knowledge received: {0}'.format(len(masks)))
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
            mask_rewards_dict[np.array2string(shell_tasks[shell_task_counter]['task_label'], precision=2, separator=', ', suppress_small=True)] = np.mean(agent.iteration_rewards)
            print(mask_rewards_dict)


        # evaluation block
        if (agent.config.eval_interval is not None and \
            shell_iterations % agent.config.eval_interval == 0):
            logger.info('*****agent {0} / evaluation block'.format(agent_id))
            _task_ids = shell_task_ids
            _tasks = shell_tasks
            _names = [eval_task_info['name'] for eval_task_info in _tasks]
            logger.info('eval tasks: {0}'.format(', '.join(_names)))
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
            print()
            task_counter_ = shell_task_counter
            logger.info('*****agent {0} / end of training on task {1}'.format(agent_id, task_counter_))
            agent.task_train_end()

            task_counter_ += 1
            shell_task_counter = task_counter_
            if task_counter_ < len(shell_tasks):
                # new task
                logger.info('*****agent {0} / set next task {1}'.format(agent_id, task_counter_))
                logger.info('task: {0}'.format(shell_tasks[task_counter_]['task']))
                logger.info('task_label: {0}'.format(shell_tasks[task_counter_]['task_label']))
                states_ = agent.task.reset_task(shell_tasks[task_counter_]) # set new task
                agent.states = agent.config.state_normalizer(states_)
                agent.task_train_start(shell_tasks[task_counter_]['task_label'])

                # set message (task_label) that will be sent to other agent as a request for
                # task knowledge (mask) about current task. this will be sent in the next
                # receive/send request cycle.
                logger.info('*****agent {0} / query other agents using current task label'\
                    .format(agent_id))
                msg = shell_tasks[task_counter_]['task_label']


                



                await_response = [True,] * num_agents
                del states_
                print()
            else:
                shell_done = True # training done for all task for agent
                logger.info('*****agent {0} / end of all training'.format(agent_id))
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

        if shell_done:
            break
        comm.barrier()
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
    

'''
ShELL distributed trainer using multiprocessing communication. Multiprocessing agent to be
added in the future.
'''
import multiprocess as mp

def shell_dist_train_mp(agent, agent_id, num_agents, task_label_dim, model_mask_dim, logger, init_address, init_port):
    logger = agent.config.logger

    # set up communication (transfer module)
    comm = ParallelizedComm(agent_id, num_agents, task_label_dim, \
        model_mask_dim, logger, init_address, init_port)
    comm.init_dist()


    # Threshold for embedding/tasklabel distance (similarity)
    THRESHOLD = 0

    logger.info('*****start shell training')

    shell_done = False
    shell_iterations = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    shell_task_ids = agent.config.task_ids
    shell_task_counter = 0

    # Initialize dictionary to store the most up-to-date rewards for an agent.
    mask_rewards_dict = dict()

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
    logger.info('*****agent {0} / setting first task (task 0)'.format(agent_id))
    logger.info('task: {0}'.format(shell_tasks[0]['task']))
    logger.info('task_label: {0}'.format(shell_tasks[0]['task_label']))
    agent.task_train_start(shell_tasks[0]['task_label'])
    print()
    del states_

    #parent, child = mp.Pipe()


    # Msg can be embedding or task label.
    msg = None
    best_agent_id = None
    # Store list of agent IDs that this agent has sent meta data to
    expecting = list()

    while True:
        start = time.time()
        # There will only be requests if there is a task change.
        # Most of the time the msg will be None
        # so nothing will happen
        other_agents_request = comm.send_receive_request(msg)
        print('SEND RECEIVE REQUEST: ', time.time() - start)
        msg = None # reset message


        ####################### SEND THE META DATA TO OTHER AGENT REQUESTS #######################
        # Send response with the mask reward, mask=None, and embedding/tasklabel

        # Go through each request from the network of agents
        requests = []
        for req in other_agents_request:
            if req is None: continue
            # Compute embedding distances between requested label and stored labels
            # i.e., subtract the requested embedding/tasklabel with the ones stored in the agent

            # If this agent has not learned anything yet, then respond with nothing
            if not mask_rewards_dict:
                req['mask_reward'] = None
                req['dist'] = None
                continue

            # If the agent has learned something (however useless it might be), check if the data is of
            # use to the requester.

            # For each embedding/tasklabel (i.e., task) learned by the agent
            # check if the embedding/tasklabel is similar enough (using THRESHOLD) to justify
            # sending back to requester.
            for key, val in enumerate(mask_rewards_dict.items()):
                dist = abs(np.sum(np.subtract(req['task_label'], np.array(key))))
                
                # If similarity is good enough send the task reward and embedding/tasklabel distance.
                # i.e., SEND META DATA TO REQUESTER

                if dist <= THRESHOLD:
                    req['mask_reward'] = val
                    req['dist'] = dist

                    # Append the requester agent id. We use this to listen for a mask request
                    # or to get rejected.
                    expecting.append(req['sender_agent_id'])

                # Otherwise send nothing
                else:
                    req['mask_reward'] = None
                    req['dist'] = None

            # Set mask to none and send the mask reward
            #req['mask'] = None
            #req['mask_reward'] = mask_rewards_dict[tuple(req['task_label'])]
            # Task label/Embedding is in req['task_label']
            requests.append(req)

        # Do a check and send out the responses to request
        if len(requests) > 0:
            comm.send_meta_response(requests)
        ####################### SEND THE META DATA TO OTHER AGENT REQUESTS #######################







        ####################### RECEIVE THE META DATA FOR THIS AGENTS REQUESTS #######################
        # receive meta data response from other agents for a embedding/tasklabel request from 
        # this agent.

        send_msk_request = dict()
        selected = False

        # Listen for any responses from other agents (containing the metadata)
        # if the agent earlier sent a request, check whether response has been sent.
        if any(await_response):
            logger.info('awaiting response: {0}'.format(await_response))
            results = comm.receive_meta_response()

            # Sort received meta data by smallest distance (primary) and highest reward (secondary),
            # using full bidirectional multikey sorting (fancy words for such a simple concept)
            results = sorted(results, key=lambda d: (d['dist'], -d['mask_reward']))

            # Iterate through the await_response list. Upon task change this is an array
            # [True,] * num_agents
            for idx in range(len(await_response)):
                # Do some checks to remove to useless results
                if await_response[idx] is False: continue
                if results[idx] is False: continue
                elif results[idx] is None: await_response[idx] = False

                # Otherwise unpack the metadata
                # At this stage we want to make a mapping between the requested embedding/tasklabel
                # and the received embeddings from metadata. This will make it easier to 
                else:
                    recv_agent_id = results[idx]['agent_id']
                    recv_msk_rw = results[idx]['mask_reward']
                    recv_label = results[idx]['emb_label']
                    recv_dist = results[idx]['dist']
                    if recv_dist <= THRESHOLD and selected == False:
                        send_msk_request[recv_agent_id] = recv_label
                        # Make a note of the best agent id in memory of this agent
                        # We will use this later to check the response from the best agent
                        best_agent_id = recv_agent_id
                        selected = True

                    else:
                        send_msk_request[recv_agent_id] = None

                    await_response[idx] = False

            # Need to fix this logging message
            logger.info('meta data received from other agents')

        ####################### RECEIVE THE META DATA FOR THIS AGENTS REQUESTS #######################



        ####################### SEND A REQUEST FOR THE MASKS #######################
        # Send a response back to each agent that sent this agent metadata. Tell them to either
        # send mask or move on.

        # Also receive the same response from other agents. If other agents want a mask from this
        # agent, then msk_requests will be populated with a dictionary for each request.
        if send_msk_request:
            msk_requests = comm.send_receive_mask_request(send_msk_request, expecting)

        # Now the agent needs to send a mask to each agent in the msk_requests list
        # Check if not empty
        if msk_requests:
            # Iterate through he requests
            for req in msk_requests:
                # Unpack the request
                agent_id = req['requester_agent_id']
                mask = agent.label_to_mask(req['task_label'])
                comm.send_mask_response(agent_id, mask)

        # Expecting a response from the best agent id
        if best_agent_id:

            # We want to get the mask from the best agent
            received_mask = comm.receive_mask_response(best_agent_id)

            agent.distil_task_knowledge_single(received_mask)

            # Reset the best agent id for the next request
            best_agent_id = None

        ####################### COMMUNICATE THE MASKS END #######################
        



                    
        # agent iteration (training step): collect on policy data and optimise agent
        dict_logs = agent.iteration()
        shell_iterations += 1

        # tensorboard log
        if shell_iterations % agent.config.iteration_log_interval == 0:
            itr_log_fn(logger, agent, agent_id, shell_iterations, shell_task_counter, dict_logs)
            
            # Create a dictionary to store the most recent iteration rewards for a mask. Update in every iteration
            # logging cycle. Take average of all worker averages as the most recent reward score for a given task
            mask_rewards_dict[tuple(shell_tasks[shell_task_counter]['task_label'])] = np.mean(agent.iteration_rewards)
            print(mask_rewards_dict)


        # evaluation block
        if (agent.config.eval_interval is not None and \
            shell_iterations % agent.config.eval_interval == 0):
            logger.info('*****agent {0} / evaluation block'.format(agent_id))
            _task_ids = shell_task_ids
            _tasks = shell_tasks
            _names = [eval_task_info['name'] for eval_task_info in _tasks]
            logger.info('eval tasks: {0}'.format(', '.join(_names)))
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
        # i.e., Task Change occurs here. For detect module, if the task embedding signifies a task
        # change then that occurs here.
        '''
        If we want to use a Fetch All mode for ShELL then we need to add a commmunication component
        at task change which broadcasts the mask to all other agents currently on the network.

        Otherwise the current implementation is a On Demand mode where each agent requests knowledge
        only when required.
        '''
        if not agent.config.max_steps: raise ValueError('`max_steps` should be set for each agent')
        task_steps_limit = agent.config.max_steps[shell_task_counter] * (shell_task_counter + 1)
        if agent.total_steps >= task_steps_limit:
            print()
            task_counter_ = shell_task_counter
            logger.info('*****agent {0} / end of training on task {1}'.format(agent_id, task_counter_))
            agent.task_train_end()

            task_counter_ += 1
            shell_task_counter = task_counter_
            if task_counter_ < len(shell_tasks):
                # new task
                logger.info('*****agent {0} / set next task {1}'.format(agent_id, task_counter_))
                logger.info('task: {0}'.format(shell_tasks[task_counter_]['task']))
                logger.info('task_label: {0}'.format(shell_tasks[task_counter_]['task_label']))
                states_ = agent.task.reset_task(shell_tasks[task_counter_]) # set new task
                agent.states = agent.config.state_normalizer(states_)
                agent.task_train_start(shell_tasks[task_counter_]['task_label'])

                # set message (task_label) that will be sent to other agent as a request for
                # task knowledge (mask) about current task. this will be sent in the next
                # receive/send request cycle.
                logger.info('*****agent {0} / query other agents using current task label'\
                    .format(agent_id))
                msg = shell_tasks[task_counter_]['task_label']
                await_response = [True,] * num_agents
                del states_
                print()
            else:
                shell_done = True # training done for all task for agent
                logger.info('*****agent {0} / end of all training'.format(agent_id))
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

        if shell_done:
            break
        comm.barrier()
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