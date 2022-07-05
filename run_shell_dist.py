#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
Shared Experience Lifelong Learning (ShELL) experiments
Multi-agent continual lifelong learners

Each agent is a ppo agent with supermask superposition 
lifelong learning algorithm.
https://arxiv.org/abs/2006.14769
'''

import json
import copy
import shutil
import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import argparse

# helper function
def global_config(config, name):
    config.env_name = name
    config.env_config_path = None
    config.lr = 0.00015
    config.cl_preservation = 'supermask'
    config.seed = 9157
    random_seed(config.seed)
    config.log_dir = None
    config.logger = None 
    config.num_workers = 4
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)

    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
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

'''
shared experience lifelong learning (ShELL)
lifelong (continual) learning algorithm for each ShELL agent: supermask superposition
RL agent/algorithm: PPO
'''
##### (Meta)CT-graph
def shell_dist_mctgraph(name, args):
    shell_config_path = args.shell_config_path
    env_config_path = args.env_config_path

    with open(shell_config_path, 'r') as f:
        shell_config = json.load(f)
        config_seed = shell_config['seed']
        shell_config = shell_config['agents'][args.agent_id]
    num_agents = args.num_agents

    # set up config
    config = Config()
    config = global_config(config, name)
    config.state_normalizer = ImageNormalizer()

    # set seed
    config.seed = config_seed
    
    # set up logging system
    exp_id = '{0}-seed-{1}'.format(args.exp_id, config.seed)
    path_name = '{0}-shell-dist-{1}/agent_{2}'.format(name, exp_id, args.agent_id)
    log_dir = get_default_log_dir(path_name)
    logger = get_logger(log_dir=log_dir, file_name='train-log')
    config.logger = logger
    config.log_dir = log_dir

    # save shell config and env config
    shutil.copy(shell_config_path, log_dir)
    shutil.copy(env_config_path, log_dir)

    # create/initialise agent
    logger.info('*****initialising agent {0}'.format(args.agent_id))
    # task may repeat, so get number of unique tasks.
    num_tasks = len(set(shell_config['task_ids'])) 
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['task_ids']
    if isinstance(shell_config['max_steps'], list):
        config.max_steps = shell_config['max_steps']
    else:
        config.max_steps = [shell_config['max_steps'], ] * num_tasks
    task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
        state_dim, action_dim, label_dim, 
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim, \
        hidden_units=(200, 200, 200), num_tasks=num_tasks), 
        actor_body=DummyBody_CL(200), 
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    # set up communication (transfer module)
    comm = Communication(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, args.init_address, args.init_port)

    # start training
    shell_dist_train(agent, comm, args.agent_id, args.num_agents)

##### Minigrid environment
def shell_dist_minigrid(name, args):
    shell_config_path = args.shell_config_path
    env_config_path = args.env_config_path

    with open(shell_config_path, 'r') as f:
        shell_config = json.load(f)
        config_seed = shell_config['seed']
        shell_config = shell_config['agents'][args.agent_id]
    num_agents = args.num_agents

    # set up config
    config = Config()
    config = global_config(config, name)
    # rescale state normaliser: suitable for grid encoding of states in minigrid
    config.state_normalizer = RescaleNormalizer(1./10.)

    # set seed
    config.seed = config_seed
    
    # set up logging system
    exp_id = '{0}-seed-{1}'.format(args.exp_id, config.seed)
    path_name = '{0}-shell-dist-{1}/agent_{2}'.format(name, exp_id, args.agent_id)
    log_dir = get_default_log_dir(path_name)
    logger = get_logger(log_dir=log_dir, file_name='train-log')
    config.logger = logger
    config.log_dir = log_dir

    # save shell config and env config
    shutil.copy(shell_config_path, log_dir)
    shutil.copy(env_config_path, log_dir)

    # create/initialise agent
    logger.info('*****initialising agent {0}'.format(args.agent_id))
    # task may repeat, so get number of unique tasks.
    num_tasks = len(set(shell_config['task_ids'])) 
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['task_ids']
    if isinstance(shell_config['max_steps'], list):
        config.max_steps = shell_config['max_steps']
    else:
        config.max_steps = [shell_config['max_steps'], ] * num_tasks
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

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    # set up communication (transfer module)
    comm = Communication(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, args.init_address, args.init_port)

    # start training
    shell_dist_train(agent, comm, args.agent_id, args.num_agents)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id', help='rank: the process id or machine id of the agent', type=int)
    parser.add_argument('num_agents', help='world: total number of agents', type=int)
    parser.add_argument('--shell_config_path', help='shell config', default='./shell.json')
    parser.add_argument('--env_name', help='name of the evaluation environment. ' \
        'minigrid and ctgraph currently supported', default='minigrid')
    parser.add_argument('--env_config_path', help='environment config', \
        default='./env_configs/minigrid_sc_3.json')
    parser.add_argument('--init_address', help='address of the master/first agent (rank/id 0) '\
        'in the pool of agents', default='127.0.0.1', type=str)
    parser.add_argument('--init_port', help='port number of the master/first agent (rank/id 0) '\
        'in the pool of agents', default='5283', type=str)
    parser.add_argument('--exp_id', help='id of the experiment (e.g, seed). useful for setting '\
        'up structured directory of experiment results/data', default='upz', type=str)
    args = parser.parse_args()

    if args.env_name == 'minigrid':
        name = 'Minigrid'
        shell_dist_minigrid(name, args)
    elif args.env_name == 'ctgraph':
        name = 'MetaCTgraph'
        shell_dist_mctgraph(name, args)
    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))
