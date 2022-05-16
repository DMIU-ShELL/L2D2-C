#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
Shared Experience Lifelong Learning (ShELL) experiments
Multi-agent continual lifelong learners
'''

import json
import copy
import shutil
import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# helper function
def global_config(config, name):
    config.env_name = name
    config.env_config_path = None
    config.lr = 0.00015
    config.cl_preservation = 'ss'
    config.seed = 9157
    random_seed(config.seed)
    config.log_dir = None
    config.logger = None 
    config.num_workers = 4
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)

    config.policy_fn = SamplePolicy
    #config.state_normalizer = ImageNormalizer()
    config.state_normalizer = RescaleNormalizer(1./10.)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1 #0.75
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = 1e3
    config.evaluation_episodes = 10
    config.cl_requires_task_label = True
    config.task_fn = None
    config.eval_task_fn = None
    config.network_fn = None 
    config.eval_interval = 25
    return config

# shared experience lifelong learning (ShELL)
# continual learning algorithm for each ShELL agent: supermask superposition
# RL agent/algorithm: PPO
def shell_minigrid(name, shell_config_path, env_config_path):
    with open(shell_config_path, 'r') as f:
        shell_config = json.load(f)
    agents = []
    num_agents = len(shell_config['agents'])
    # set up logging system
    exp_id = ''
    log_dir = get_default_log_dir(name + '-shell' + exp_id)
    logger = get_logger(log_dir=log_dir, file_name='train-log')
    # create/initialise agents
    for idx in range(num_agents):
        logger.info('*****initialising agent {0}'.format(idx))
        config = Config()
        config = global_config(config, name)
        # task may repeat, so get number of unique tasks.
        num_tasks = len(set(shell_config['agents'][idx]['task_ids'])) 
        config.cl_num_tasks = num_tasks
        config.task_ids = shell_config['agents'][idx]['task_ids']
        if isinstance(shell_config['agents'][idx]['max_steps'], list):
            config.max_steps = shell_config['agents'][idx]['max_steps']
        else:
            config.max_steps = [shell_config['agents'][idx]['max_steps'], ] * num_tasks
        task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, False)
        config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir)
        eval_task_fn= lambda log_dir: MiniGridFlatObs(name, env_config_path,log_dir,config.seed,True)
        config.eval_task_fn = eval_task_fn
        config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
            state_dim, action_dim, label_dim, 
            phi_body=FCBody_SS(state_dim, task_label_dim=label_dim, \
            hidden_units=(200, 200, 200), num_tasks=num_tasks), 
            actor_body=DummyBody_CL(200), 
            critic_body=DummyBody_CL(200),
            num_tasks=num_tasks)

        agent = PPOAgentSS(config)
        config.agent_name = agent.__class__.__name__ + '_{0}'.format(idx)
        agents.append(agent)

    shell_train(agents, logger)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    # minigrid experiments
    game = 'MiniGrid'

    parser = argparse.ArgumentParser()
    parser.add_argument('--shell_config_path', help='shell config', default='./shell.json')
    parser.add_argument('--env_config_path',help='environment config',default='./minigrid_sc_3.json')
    args = parser.parse_args()

    shell_minigrid(game, args.shell_config_path, args.env_config_path)
