import numpy as np
import pickle
import os
import datetime
import torch
from .torch_utils import *

#run iterations continual learning setting using supermask superposition (ss) algorithm
def run_iterations_w_oracle(agent, tasks_info):
    mod_rm = ResourceManager()
    mod_detect = Detect()

    config = agent.config

    log_path_tstats = config.log_dir + '/task_stats'
    if not os.path.exists(log_path_tstats):
        os.makedirs(log_path_tstats)
    log_path_eval = config.log_dir + '/eval'
    if not os.path.exists(log_path_eval):
        os.makedirs(log_path_eval)
    random_seed(config.seed)
    agent_name = agent.__class__.__name__

    iteration = 0
    steps = []
    rewards = []
    task_start_idx = 0
    num_tasks = len(tasks_info)

    eval_tracker = False
    eval_data = []
    metric_tcr = [] # tcr => total cumulative reward

    task_change = False
    for learn_block_idx in range(config.cl_num_learn_blocks):
        config.logger.info('********** start of learning block {0}'.format(learn_block_idx))
        eval_results = {task_idx:[] for task_idx in range(len(tasks_info))}

        for task_idx, task_info in enumerate(tasks_info):
            config.logger.info('*****start training on task {0}'.format(task_idx))
            config.logger.info('task: {0}'.format(task_info['task']))

            states = agent.task.reset_task(task_info)
            agent.states = config.state_normalizer(states)
            if task_idx == 0:
                # boostrap learning w/o the detect module for only the first task.
                agent.set_first_task(task_info['task_label'], task_info['name'])
            while True:
                # train step
                bool_execute = mod_rm.operation(ResourceManager.OP_ID_TRAIN)
                if bool_execute:
                    avg_grad_norm = agent.iteration()
                iteration += 1
                steps.append(agent.total_steps)
                rewards.append(np.mean(agent.last_episode_rewards))

                # detect task
                bool_execute = mod_rm.operation(ResourceManager.OP_ID_DETECT)
                if bool_execute:
                    task_label, task_change = mod_detect.detect(agent.data_buffer)
                if task_change and task_idx > 0:
                    config.logger.info('*****task change detected by agent')
                    config.logger.info('cacheing mask for current task')
                    agent.task_change_detected(task_label, task_info['name'])
                    task_change = False
                else:
                    agent.update_task_label(task_label)

                # logging
                if iteration % config.iteration_log_interval == 0:
                    config.logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f'%(
                        iteration, agent.total_steps, np.mean(agent.last_episode_rewards),
                        np.max(agent.last_episode_rewards),
                        np.min(agent.last_episode_rewards)
                    ))
                    config.logger.scalar_summary('avg reward', np.mean(agent.last_episode_rewards))
                    config.logger.scalar_summary('max reward', np.max(agent.last_episode_rewards))
                    config.logger.scalar_summary('min reward', np.min(agent.last_episode_rewards))
                    config.logger.scalar_summary('avg grad norm', avg_grad_norm)

                    with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % \
                        (agent_name, config.tag, agent.task.name), 'wb') as f:
                        pickle.dump({'rewards': rewards, 'steps': steps}, f)
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    for tag, value in agent.network.named_parameters():
                        tag = tag.replace('.', '/')
                        config.logger.histo_summary(tag, value.data.cpu().numpy())
                    if hasattr(agent, 'layers_output'):
                        for tag, value in agent.layers_output:
                            tag = 'layer_output/' + tag
                            config.logger.histo_summary(tag, value.data.cpu().numpy())

                # evaluation block
                bool_execute = mod_rm.operation(ResourceManager.OP_ID_EVAL)
                if bool_execute:
                    if (agent.config.eval_interval is not None and \
                        iteration % agent.config.eval_interval == 0):
                        config.logger.info('*****agent / evaluation block')
                        _tasks = tasks_info
                        _names = [eval_task_info['name'] for eval_task_info in _tasks]
                        config.logger.info('eval tasks: {0}'.format(', '.join(_names)))
                        eval_data.append(np.zeros(len(_tasks),))
                        for eval_task_idx, eval_task_info in enumerate(_tasks):
                            agent.task_eval_start(eval_task_info['name'])
                            eval_states = agent.evaluation_env.reset_task(eval_task_info)
                            agent.evaluation_states = eval_states
                            rewards, _ = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                            agent.task_eval_end()
                            eval_data[-1][eval_task_idx] = np.mean(rewards)
                        tcr = eval_data[-1].sum()
                        metric_tcr.append(tcr)
                        tp = np.sum(metric_tcr)
                        config.logger.info('*****cl evaluation:')
                        config.logger.info('cl eval TCR: {0}'.format(tcr))
                        config.logger.info('cl eval TP: {0}'.format(tp))
                        config.logger.scalar_summary('cl_eval/tcr', tcr)
                        config.logger.scalar_summary('cl_eval/tp', np.sum(metric_tcr))

                # check whether task training is done
                task_steps_limit = config.max_steps * (num_tasks * learn_block_idx + task_idx + 1)
                if config.max_steps and agent.total_steps >= task_steps_limit:
                    with open(log_path_tstats + '/%s-%s-online-stats-%s-run-%d-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                        pickle.dump({'rewards': rewards[task_start_idx : ], \
                        'steps': steps[task_start_idx : ]}, f)
                    agent.save(log_path_tstats +'/%s-%s-model-%s-run-%d-task-%d.bin' % (agent_name, \
                        config.tag, agent.task.name, learn_block_idx+1, task_idx+1))
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    task_start_idx = len(rewards)
                    break
            # end of current task training
            # final evaluation block after completing traning on a task (for debugging purpose)
            # only evaluate agent across task exposed to agent so far
            config.logger.info('evaluating agent across all tasks exposed so far to agent')
            for j in range(task_idx+1):
                _eval_task = tasks_info[j]
                agent.task_eval_start(_eval_task['name'])

                eval_states = agent.evaluation_env.reset_task(tasks_info[j])
                agent.evaluation_states = eval_states
                rewards, episodes = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                eval_results[j] += rewards

                agent.task_eval_end()

                with open(config.log_dir+'/rewards-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(rewards, f)
                with open(config.log_dir+'/episodes-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(episodes, f)
        print('eval stats')
        with open(log_path_eval + '/eval_full_stats.bin', 'wb') as f: pickle.dump(eval_results, f)

        f = open(log_path_eval + '/eval_stats.csv', 'w')
        f.write('task_id,avg_reward\n')
        for k, v in eval_results.items():
            print('{0}: {1:.4f}'.format(k, np.mean(v)))
            f.write('{0},{1:.4f}\n'.format(k, np.mean(v)))
            config.logger.scalar_summary('zeval/task_{0}/avg_reward'.format(k), np.mean(v))
        f.close()
        config.logger.info('********** end of learning block {0}\n'.format(learn_block_idx))

    to_save = np.stack(eval_data, axis=0)
    with open(log_path_eval + '/eval_metrics.npy', 'wb') as f:
        np.save(f, to_save)
    agent.close()
    return steps, rewards

# run iterations 
# either baseline with no lifelong learning or
# continual learning setting using supermask superposition (ss)
# algorithm without detect and resource manager modules turned on.
def run_iterations_wo_oracle(agent, tasks_info):
    config = agent.config

    log_path_tstats = config.log_dir + '/task_stats'
    if not os.path.exists(log_path_tstats):
        os.makedirs(log_path_tstats)
    log_path_eval = config.log_dir + '/eval'
    if not os.path.exists(log_path_eval):
        os.makedirs(log_path_eval)
    random_seed(config.seed)
    agent_name = agent.__class__.__name__

    iteration = 0
    steps = []
    rewards = []
    task_start_idx = 0
    num_tasks = len(tasks_info)

    eval_tracker = False
    eval_data = []
    metric_tcr = [] # tcr => total cumulative reward

    for learn_block_idx in range(config.cl_num_learn_blocks):
        config.logger.info('********** start of learning block {0}'.format(learn_block_idx))
        eval_results = {task_idx:[] for task_idx in range(len(tasks_info))}

        for task_idx, task_info in enumerate(tasks_info):
            config.logger.info('*****start training on task {0}'.format(task_idx))
            config.logger.info('task: {0}'.format(task_info['task']))
            config.logger.info('task_label: {0}'.format(task_info['task_label']))

            states = agent.task.reset_task(task_info)
            agent.states = config.state_normalizer(states)
            agent.data_buffer.clear()
            agent.task_train_start(task_info['task_label'])
            while True:
                # train step
                avg_grad_norm = agent.iteration()
                iteration += 1
                steps.append(agent.total_steps)
                rewards.append(np.mean(agent.last_episode_rewards))

                # logging
                if iteration % config.iteration_log_interval == 0:
                    config.logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f'%(
                        iteration, agent.total_steps, np.mean(agent.last_episode_rewards),
                        np.max(agent.last_episode_rewards),
                        np.min(agent.last_episode_rewards)
                    ))
                    config.logger.scalar_summary('avg reward', np.mean(agent.last_episode_rewards))
                    config.logger.scalar_summary('max reward', np.max(agent.last_episode_rewards))
                    config.logger.scalar_summary('min reward', np.min(agent.last_episode_rewards))
                    config.logger.scalar_summary('avg grad norm', avg_grad_norm)

                    with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % \
                        (agent_name, config.tag, agent.task.name), 'wb') as f:
                        pickle.dump({'rewards': rewards, 'steps': steps}, f)
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    for tag, value in agent.network.named_parameters():
                        tag = tag.replace('.', '/')
                        config.logger.histo_summary(tag, value.data.cpu().numpy())
                    if hasattr(agent, 'layers_output'):
                        for tag, value in agent.layers_output:
                            tag = 'layer_output/' + tag
                            config.logger.histo_summary(tag, value.data.cpu().numpy())

                # evaluation block
                if (agent.config.eval_interval is not None and \
                    iteration % agent.config.eval_interval == 0):
                    config.logger.info('*****agent / evaluation block')
                    _tasks = tasks_info
                    _names = [eval_task_info['task'] for eval_task_info in _tasks]
                    config.logger.info('eval tasks: {0}'.format(', '.join(_names)))
                    eval_data.append(np.zeros(len(_tasks),))
                    for eval_task_idx, eval_task_info in enumerate(_tasks):
                        agent.task_eval_start(eval_task_info['task_label'])
                        eval_states = agent.evaluation_env.reset_task(eval_task_info)
                        agent.evaluation_states = eval_states
                        rewards, _ = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                        agent.task_eval_end()
                        eval_data[-1][eval_task_idx] = np.mean(rewards)
                    tcr = eval_data[-1].sum()
                    metric_tcr.append(tcr)
                    tp = np.sum(metric_tcr)
                    config.logger.info('*****cl evaluation:')
                    config.logger.info('cl eval TCR: {0}'.format(tcr))
                    config.logger.info('cl eval TP: {0}'.format(tp))
                    config.logger.scalar_summary('cl_eval/tcr', tcr)
                    config.logger.scalar_summary('cl_eval/tp', np.sum(metric_tcr))


                # check whether task training has been completed
                task_steps_limit = config.max_steps * (num_tasks * learn_block_idx + task_idx + 1)
                if config.max_steps and agent.total_steps >= task_steps_limit:
                    with open(log_path_tstats + '/%s-%s-online-stats-%s-run-%d-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                        pickle.dump({'rewards': rewards[task_start_idx : ], \
                        'steps': steps[task_start_idx : ]}, f)
                    agent.save(log_path_tstats +'/%s-%s-model-%s-run-%d-task-%d.bin' % (agent_name, \
                        config.tag, agent.task.name, learn_block_idx+1, task_idx+1))
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    task_start_idx = len(rewards)
                    break

            # end of current task training
            if agent_name != 'BaselineAgent':
                config.logger.info('cacheing mask for current task')
            ret = agent.task_train_end()
            # evaluate agent across task exposed to agent so far
            config.logger.info('evaluating agent across all tasks exposed so far to agent')
            for j in range(task_idx+1):
                _eval_task = tasks_info[j]
                agent.task_eval_start(_eval_task['task_label'])

                eval_states = agent.evaluation_env.reset_task(tasks_info[j])
                agent.evaluation_states = eval_states
                rewards, episodes = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                eval_results[j] += rewards

                agent.task_eval_end()

                with open(config.log_dir+'/rewards-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(rewards, f)
                with open(config.log_dir+'/episodes-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(episodes, f)
        print('eval stats')
        with open(log_path_eval + '/eval_full_stats.bin', 'wb') as f: pickle.dump(eval_results, f)

        f = open(log_path_eval + '/eval_stats.csv', 'w')
        f.write('task_id,avg_reward\n')
        for k, v in eval_results.items():
            print('{0}: {1:.4f}'.format(k, np.mean(v)))
            f.write('{0},{1:.4f}\n'.format(k, np.mean(v)))
            config.logger.scalar_summary('zeval/task_{0}/avg_reward'.format(k), np.mean(v))
        f.close()
        config.logger.info('********** end of learning block {0}\n'.format(learn_block_idx))

    if len(eval_data) > 0:
        to_save = np.stack(eval_data, axis=0)
        with open(log_path_eval + '/eval_metrics.npy', 'wb') as f:
            np.save(f, to_save)
    agent.close()
    return steps, rewards
