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
from ..shell_modules import *

try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path


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
    shell_task_idx = [0,] * num_agents

    shell_eval_tracker = [False,] * num_agents
    shell_eval_data = []
    num_eval_tasks = len(agents[0].evaluation_env.get_all_tasks())
    shell_eval_data.append(np.zeros((num_agents, num_eval_tasks), dtype=np.float32))
    shell_metric_tcr = [] # tcr => total cumulative reward metric

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
            agent.iteration()
            shell_iterations[agent_idx] += 1
            # tensorboard log
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

            # evaluation block
            if (agent.config.eval_interval is not None and \
                shell_iterations[agent_idx] % agent.config.eval_interval == 0):
                logger.info('*****agent {0} / evaluation block'.format(agent_idx))
                _tasks = agent.evaluation_env.get_all_tasks()
                _names = [eval_task_info['task'] for eval_task_info in _tasks]
                logger.info('eval tasks: {0}'.format(', '.join(_names)))
                for eval_task_idx, eval_task_info in enumerate(_tasks):
                    agent.task_eval_start(eval_task_info['task_label'])
                    eval_states = agent.evaluation_env.reset_task(eval_task_info)
                    agent.evaluation_states = eval_states
                    rewards, _ = agent.evaluate_cl(num_iterations=agent.config.evaluation_episodes)
                    agent.task_eval_end()
                    shell_eval_data[-1][agent_idx, eval_task_idx] = np.mean(rewards)
                shell_eval_tracker[agent_idx] = True

            # checker for end of task training
            if not agent.config.max_steps:
                raise ValueError('`max_steps` should be set for each agent')
            task_steps_limit = agent.config.max_steps[shell_task_idx[agent_idx]] * \
                (shell_task_idx[agent_idx] + 1)
            if agent.total_steps >= task_steps_limit:
                print()
                task_idx_ = shell_task_idx[agent_idx]
                logger.info('*****agent {0} / end of training on task {1}'.format(agent_idx, \
                    task_idx_))
                #agent.task_train_end(shell_tasks[agent_idx][task_idx_]['task_label'])
                agent.task_train_end()

                task_idx_ += 1
                shell_task_idx[agent_idx] = task_idx_
                if task_idx_ < len(shell_tasks[agent_idx]):
                    logger.info('*****agent {0} / set next task {1}'.format(agent_idx, task_idx_))
                    logger.info('task: {0}'.format(shell_tasks[agent_idx][task_idx_]['task']))
                    logger.info('task_label: {0}'.format(shell_tasks[agent_idx][task_idx_]['task_label']))
                    states_ = agent.task.reset_task(shell_tasks[agent_idx][task_idx_]) # set new task
                    agent.states = agent.config.state_normalizer(states_)
                    agent.task_train_start(shell_tasks[agent_idx][task_idx_]['task_label'])

                    # ping other agents to see if they have knoweledge (mask) of current task
                    logger.info('*****agent {0} / pinging other agents to leverage on existing ' \
                        'knowledge about task'.format(agent_idx))
                    a_idxs = list(range(len(agents)))
                    a_idxs.remove(agent_idx)
                    other_agents = [agents[i] for i in a_idxs]
                    found_knowledge = agent.ping_agents(other_agents)
                    if found_knowledge:
                        logger.info('found knowledge about task from other agents')
                    else:
                        logger.info('could not find any agent with knowledge about task')
                    del states_
                    print()
                else:
                    shell_done[agent_idx] = True # training done for all task for agent
                    logger.info('*****agent {0} / end of all training'.format(agent_idx))
                del task_idx_
                    
        if all(shell_eval_tracker):
            _metrics = shell_eval_data[-1]
            # compute tcr 
            _max_reward = _metrics.max(axis=0) 
            _agent_ids = _metrics.argmax(axis=0).tolist()
            _agent_ids = ', '.join([str(_agent_id) for _agent_id in _agent_ids])
            tcr = _max_reward.sum()
            shell_metric_tcr.append(tcr)
            # log eval to file/screen and tensorboard
            logger.info('*****shell evaluation:')
            logger.info('best agent per task:'.format(_agent_ids))
            logger.info('shell eval TCR: {0}'.format(tcr))
            logger.info('shell eval TP: {0}'.format(np.sum(shell_metric_tcr)))
            logger.scalar_summary('shell_eval/tcr', tcr)
            logger.scalar_summary('shell_eval/tp', np.sum(shell_metric_tcr))
            # reset eval tracker
            shell_eval_tracker = [False for _ in shell_eval_tracker]
            # initialise new eval block
            shell_eval_data.append(np.zeros((num_agents, num_eval_tasks), dtype=np.float32))

        if all(shell_done):
            break
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
    shell_done = False
    shell_iterations = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    shell_task_idx = 0

    shell_eval_tracker = False
    shell_eval_data = []
    num_eval_tasks = len(agent.evaluation_env.get_all_tasks())
    shell_eval_data.append(np.zeros((1, num_eval_tasks), dtype=np.float32))
    shell_metric_tcr = [] # tcr => total cumulative reward metric

    await_response = [False,] * num_agents # flag

    logger = agent.config.logger

    print()
    logger.info('*****start shell training')

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
        agent.iteration()
        shell_iterations += 1

        # listen for and send info (task label or NULL message) to other agents
        other_agents_request = comm.send_receive_request(msg)
        msg = None # reset message

        requests = []
        for req in other_agents_request:
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
            # TODO still need to fix how mask is used.
            agent.distil_task_knowledge(masks)
                    
        # tensorboard log
        if shell_iterations % agent.config.iteration_log_interval == 0:
            logger.info('agent %d, task %d / iteration %d, total steps %d, ' \
            'mean/max/min reward %f/%f/%f' % (agent_id, shell_task_idx, \
                shell_iterations,
                agent.total_steps,
                np.mean(agent.last_episode_rewards),
                np.max(agent.last_episode_rewards),
                np.min(agent.last_episode_rewards)
            ))
            logger.scalar_summary('agent_{0}/avg_reward'.format(agent_id), \
                np.mean(agent.last_episode_rewards))
            logger.scalar_summary('agent_{0}/max_reward'.format(agent_id), \
                np.max(agent.last_episode_rewards))
            logger.scalar_summary('agent_{0}/min_reward'.format(agent_id), \
                np.min(agent.last_episode_rewards))

        # evaluation block
        if (agent.config.eval_interval is not None and \
            shell_iterations % agent.config.eval_interval == 0):
            logger.info('*****agent {0} / evaluation block'.format(agent_id))
            _tasks = agent.evaluation_env.get_all_tasks()
            _names = [eval_task_info['task'] for eval_task_info in _tasks]
            logger.info('eval tasks: {0}'.format(', '.join(_names)))
            for eval_task_idx, eval_task_info in enumerate(_tasks):
                agent.task_eval_start(eval_task_info['task_label'])
                eval_states = agent.evaluation_env.reset_task(eval_task_info)
                agent.evaluation_states = eval_states
                rewards, _ = agent.evaluate_cl(num_iterations=agent.config.evaluation_episodes)
                agent.task_eval_end()
                shell_eval_data[-1][0, eval_task_idx] = np.mean(rewards)
            shell_eval_tracker = True

        # end of current task training. move onto next task or end training if last task.
        if not agent.config.max_steps: raise ValueError('`max_steps` should be set for each agent')
        task_steps_limit = agent.config.max_steps[shell_task_idx] * (shell_task_idx + 1)
        if agent.total_steps >= task_steps_limit:
            print()
            task_idx_ = shell_task_idx
            logger.info('*****agent {0} / end of training on task {1}'.format(agent_id, task_idx_))
            agent.task_train_end()

            task_idx_ += 1
            shell_task_idx = task_idx_
            if task_idx_ < len(shell_tasks):
                # new task
                logger.info('*****agent {0} / set next task {1}'.format(agent_id, task_idx_))
                logger.info('task: {0}'.format(shell_tasks[task_idx_]['task']))
                logger.info('task_label: {0}'.format(shell_tasks[task_idx_]['task_label']))
                states_ = agent.task.reset_task(shell_tasks[task_idx_]) # set new task
                agent.states = agent.config.state_normalizer(states_)
                agent.task_train_start(shell_tasks[task_idx_]['task_label'])

                # set message (task_label) that will be sent to other agent as a request for
                # task knowledge (mask) about current task. this will be sent in the next
                # receive/send request cycle.
                logger.info('*****agent {0} / query other agents using current task label'\
                    .format(agent_id))
                msg = shell_tasks[task_idx_]['task_label']
                await_response = [True,] * num_agents
                del states_
                print()
            else:
                shell_done = True # training done for all task for agent
                logger.info('*****agent {0} / end of all training'.format(agent_id))
            del task_idx_
                    
        #if all(shell_eval_tracker):
        #    _metrics = shell_eval_data[-1]
        #    # compute tcr 
        #    _max_reward = _metrics.max(axis=0) 
        #    _agent_ids = _metrics.argmax(axis=0).tolist()
        #    _agent_ids = ', '.join([str(_agent_id) for _agent_id in _agent_ids])
        #    tcr = _max_reward.sum()
        #    shell_metric_tcr.append(tcr)
        #    # log eval to file/screen and tensorboard
        #    logger.info('*****shell evaluation:')
        #    logger.info('best agent per task:'.format(_agent_ids))
        #    logger.info('shell eval TCR: {0}'.format(tcr))
        #    logger.info('shell eval TP: {0}'.format(np.sum(shell_metric_tcr)))
        #    logger.scalar_summary('shell_eval/tcr', tcr)
        #    logger.scalar_summary('shell_eval/tp', np.sum(shell_metric_tcr))
        #    # reset eval tracker
        #    shell_eval_tracker = [False for _ in shell_eval_tracker]
        #    # initialise new eval block
        #    shell_eval_data.append(np.zeros((1, num_eval_tasks), dtype=np.float32))

        if shell_done:
            break
        comm.barrier()
    # end of while True

    # save eval metrics
    to_save = np.stack(shell_eval_data, axis=0)
    with open(logger.log_dir + '/eval_metrics_agent_{0}.npy'.format(agent_id), 'wb') as f:
        np.save(f, to_save)

    agent.close()
    return

