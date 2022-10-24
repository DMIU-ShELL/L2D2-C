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
import multiprocess as mp
from deep_rl.utils.misc import mkdir, get_default_log_dir
from deep_rl.utils.torch_utils import set_one_thread, random_seed, select_device
from deep_rl.utils.config import Config
from deep_rl.component.policy import SamplePolicy
from deep_rl.utils.normalizer import ImageNormalizer, RescaleNormalizer, RunningStatsNormalizer, RewardRunningStatsNormalizer
from deep_rl.utils.logger import get_logger
from deep_rl.utils.trainer_shell import shell_dist_train_mp, shell_dist_eval_mp
from deep_rl.agent.PPO_agent import ShellAgent_DP
from deep_rl.shell_modules.communication.comms import ParallelComm, ParallelCommEval
from deep_rl.component.task import ParallelizedTask, MiniGridFlatObs, MetaCTgraphFlatObs, ContinualWorld
from deep_rl.network.network_heads import CategoricalActorCriticNet_SS, GaussianActorCriticNet_SS
from deep_rl.network.network_bodies import FCBody_SS, DummyBody_CL
#from deep_rl import *
import argparse
import torch



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
    config.max_steps = 512
    config.evaluation_episodes = 5
    config.cl_requires_task_label = True
    config.task_fn = None
    config.eval_task_fn = None
    config.network_fn = None 
    config.eval_interval = None#1
    return config

'''
ShELL distributed system with multiprocessing.
Currently only communication. Data collection and model optimisation to be added
in the future.
'''
##### (Meta)CT-graph
def shell_dist_mctgraph_mp(name, args, shell_config):
    shell_config_path = args.shell_config_path
    num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    init_address = shell_config['dist_only']['init_address']
    init_port = shell_config['dist_only']['init_port']

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
    num_tasks = len(set(shell_config['curriculum']['task_ids']))
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['curriculum']['task_ids']
    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        config.max_steps = [shell_config['curriculum']['max_steps'], ] * len(shell_config['curriculum']['task_ids'])

    task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim,
        hidden_units=(200, 200, 200), num_tasks=num_tasks),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    # set up communication (transfer module)
    mode = 'ondemand'
    comm = ParallelComm(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, init_address, init_port, mode)

    # start training
    shell_dist_train_mp(agent, comm, args.agent_id, args.num_agents)

def shell_dist_mctgraph_eval(name, args, shell_config):
    shell_config_path = args.shell_config_path
    num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    init_address = shell_config['dist_only']['init_address']
    init_port = shell_config['dist_only']['init_port']

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
    num_tasks = len(set(shell_config['curriculum']['task_ids']))
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['curriculum']['task_ids']
    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        config.max_steps = [shell_config['curriculum']['max_steps'], ] * len(shell_config['curriculum']['task_ids'])

    task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim,
        hidden_units=(200, 200, 200), num_tasks=num_tasks),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    # set up communication (transfer module)
    mode = 'ondemand'
    comm = ParallelComm(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, init_address, init_port, mode)

    # start training
    shell_dist_eval_mp(agent, comm, args.agent_id, args.num_agents)


##### Minigrid environment
def shell_dist_minigrid_mp(name, args, shell_config):
    shell_config_path = args.shell_config_path
    num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    init_address = shell_config['dist_only']['init_address']
    init_port = shell_config['dist_only']['init_port']

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
    num_tasks = len(set(shell_config['curriculum']['task_ids']))
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['curriculum']['task_ids']
    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        config.max_steps = [shell_config['curriculum']['max_steps'], ] * num_tasks
    task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, False)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir, single_process=False)
    eval_task_fn= lambda log_dir: MiniGridFlatObs(name, env_config_path,log_dir,config.seed,True)
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim,
        hidden_units=(200, 200, 200), num_tasks=num_tasks),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)



    mode = 'ondemand'
    comm = ParallelComm(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, init_address, init_port, mode)




    # For full parallelisation
    #agent = ParallelizedAgent(config, args.agent_id, args.num_agents)
    #config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    # set up communication (transfer module)
    #comm = ParallelizedComm(args.agent_id, args.num_agents, agent.task_label_dim(), \
    #    agent.model_mask_dim(), logger, init_address, init_port, queue_agent, queue_comm)
    #comm = None



    # start training
    shell_dist_train_mp(agent, comm, args.agent_id, args.num_agents)

def shell_dist_minigrid_eval(name, args, shell_config):
    shell_config_path = args.shell_config_path
    num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    init_address = shell_config['dist_only']['init_address']
    init_port = shell_config['dist_only']['init_port']

    # set up config
    config = Config()
    config = global_config(config, name)
    # rescale state normaliser: suitable for grid encoding of states in minigrid
    config.state_normalizer = RescaleNormalizer(1./10.)

    # set seed
    config.seed = config_seed
    
    # set up logging system
    exp_id = '{0}-seed-{1}'.format(args.exp_id, config.seed)
    path_name = '{0}-shell-eval-{1}/agent_{2}'.format(name, exp_id, args.agent_id)
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
    num_tasks = len(set(shell_config['curriculum']['task_ids']))
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['curriculum']['task_ids']
    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        config.max_steps = [shell_config['curriculum']['max_steps'], ] * len(shell_config['curriculum']['task_ids'])

    task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, False)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir, single_process=False)
    eval_task_fn= lambda log_dir: MiniGridFlatObs(name, env_config_path,log_dir,config.seed,True)
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim,
        hidden_units=(200, 200, 200), num_tasks=num_tasks),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)



    mode = 'ondemand'
    comm = ParallelCommEval(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, init_address, init_port, mode)


    # start evaluation on curriculum
    shell_dist_eval_mp(agent, comm, args.agent_id, args.num_agents)


##### ContinualWorld environment
def shell_dist_continualworld_mp(name, args, shell_config):
    shell_config_path = args.shell_config_path
    num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    init_address = shell_config['dist_only']['init_address']
    init_port = shell_config['dist_only']['init_port']

    # set up config
    config = Config()
    config = global_config(config, name)
    # set seed
    config.seed = config_seed

    #config.state_normalizer = RescaleNormalizer(1.) # no rescaling
    config.state_normalizer = RunningStatsNormalizer()
    config.reward_normalizer = RewardRunningStatsNormalizer()
    config.num_workers = 1
    config.rollout_length = 512 * 10
    config.lr = 5e-4
    config.gae_tau = 0.97
    config.entropy_weight = 5e-3
    config.optimization_epochs = 16
    config.ppo_ratio_clip = 0.2
    config.eval_interval = 200
    config.num_mini_batches = 160 # with rollout of 5120, 160 mini_batch gives 32 samples per batch
    config.evaluation_episodes = 10
    config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)

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
    num_tasks = len(set(shell_config['curriculum']['task_ids']))
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['curriculum']['task_ids']
    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        config.max_steps = [shell_config['curriculum']['max_steps'], ] * num_tasks
    task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir)
    eval_task_fn= lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: GaussianActorCriticNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=DummyBody_CL(state_dim, task_label_dim=label_dim),
        actor_body=FCBody_SS(state_dim+label_dim, hidden_units=(128, 128), \
            discrete_mask=False, gate=torch.tanh, num_tasks=num_tasks),
        critic_body=FCBody_SS(state_dim+label_dim,hidden_units=(128, 128), \
            discrete_mask=False, gate=torch.tanh, num_tasks=num_tasks),
        num_tasks=num_tasks)

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    # set up communication (transfer module)
    mode = 'ondemand'
    comm = ParallelComm(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, init_address, init_port, mode)

    # start training
    shell_dist_train_mp(agent, comm, args.agent_id, args.num_agents)

def shell_dist_continualworld_eval(name, args, shell_config):
    shell_config_path = args.shell_config_path
    num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    init_address = shell_config['dist_only']['init_address']
    init_port = shell_config['dist_only']['init_port']

    # set up config
    config = Config()
    config = global_config(config, name)
    # set seed
    config.seed = config_seed

    #config.state_normalizer = RescaleNormalizer(1.) # no rescaling
    config.state_normalizer = RunningStatsNormalizer()
    config.reward_normalizer = RewardRunningStatsNormalizer()
    config.num_workers = 1
    config.rollout_length = 512 * 10
    config.lr = 5e-4
    config.gae_tau = 0.97
    config.entropy_weight = 5e-3
    config.optimization_epochs = 16
    config.ppo_ratio_clip = 0.2
    config.eval_interval = 200
    config.num_mini_batches = 160 # with rollout of 5120, 160 mini_batch gives 32 samples per batch
    config.evaluation_episodes = 10
    config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)

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
    num_tasks = len(set(shell_config['curriculum']['task_ids']))
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['curriculum']['task_ids']
    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        config.max_steps = [shell_config['curriculum']['max_steps'], ] * num_tasks
    task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir)
    eval_task_fn= lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: GaussianActorCriticNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=DummyBody_CL(state_dim, task_label_dim=label_dim),
        actor_body=FCBody_SS(state_dim+label_dim, hidden_units=(128, 128), \
            discrete_mask=False, gate=torch.tanh, num_tasks=num_tasks),
        critic_body=FCBody_SS(state_dim+label_dim,hidden_units=(128, 128), \
            discrete_mask=False, gate=torch.tanh, num_tasks=num_tasks),
        num_tasks=num_tasks)

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    # set up communication (transfer module)
    mode = 'ondemand'
    comm = ParallelCommEval(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, init_address, init_port, mode)

    # start training
    shell_dist_eval_mp(agent, comm, args.agent_id, args.num_agents)




if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(1)

    mp.set_start_method('fork', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id', help='rank: the process id or machine id of the agent', type=int)
    parser.add_argument('num_agents', help='world: total number of agents', type=int)
    parser.add_argument('--shell_config_path', help='shell config', default='./shell.json')
    parser.add_argument('--exp_id', help='id of the experiment. useful for setting '\
        'up structured directory of experiment results/data', default='upz', type=str)
    parser.add_argument('--mode', help='indicate evaluation agent', type=int, default=0)
    args = parser.parse_args()

    print(args)

    with open(args.shell_config_path, 'r') as f:
        shell_config = json.load(f)
        shell_config['curriculum'] = shell_config['agents'][args.agent_id]
        del shell_config['agents'][args.agent_id]

    if shell_config['env']['env_name'] == 'minigrid':
        name = Config.ENV_MINIGRID
        if args.mode == 1:
            shell_dist_minigrid_eval(name, args, shell_config)
        elif args.mode == 2:
            shell_dist_minigrid_mp(name, args, shell_config)
        else:
            raise ValueError('--mode {0} not implemented'.format(args.mode))

    elif shell_config['env']['env_name'] == 'ctgraph':
        name = Config.ENV_METACTGRAPH
        if args.mode == 1:
            shell_dist_mctgraph_eval(name, args, shell_config)
        elif args.mode == 2:
            shell_dist_mctgraph_mp(name, args, shell_config)
        else:
            raise ValueError('--mode {0} not implemented'.format(args.mode))

    elif shell_config['env']['env_name'] == 'continualworld':
        name = Config.ENV_CONTINUALWORLD
        if args.mode == 1:
            shell_dist_continualworld_eval(name, args, shell_config)
        elif args.mode == 2:
            shell_dist_continualworld_mp(name, args, shell_config)
        else:
            raise ValueError('--mode {0} not implemented'.format(args.mode))

    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))