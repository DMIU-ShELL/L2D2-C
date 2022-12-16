#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


#   .____                               .__      _________            .___             
#   |    |   _____   __ __  ____   ____ |  |__   \_   ___ \  ____   __| _/____   ______
#   |    |   \__  \ |  |  \/    \_/ ___\|  |  \  /    \  \/ /  _ \ / __ |/ __ \ /  ___/
#   |    |___ / __ \|  |  /   |  \  \___|   Y  \ \     \___(  <_> ) /_/ \  ___/ \___ \ 
#   |_______ (____  /____/|___|  /\___  >___|  /  \______  /\____/\____ |\___  >____  >
#           \/    \/           \/     \/     \/          \/            \/    \/     \/ 




'''
Shared Experience Lifelong Learning (ShELL) experiments
Multi-agent continual lifelong learners

Each agent is a ppo agent with supermask superposition 
lifelong learning algorithm.
https://arxiv.org/abs/2006.14769
'''

import json
import shutil
import matplotlib
matplotlib.use("Pdf")
import multiprocessing as mp
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
import argparse
import torch
import random



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
    config.max_steps = 25600
    config.evaluation_episodes = 2#50
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
    init_address = args.ip#shell_config['dist_only']['init_address']
    init_port = args.port#shell_config['dist_only']['init_port']

    # set up config
    config = Config()
    config = global_config(config, name)
    config.state_normalizer = ImageNormalizer()

    # set seed
    config.seed = 9157#config_seed              # Chris
    
    # set up logging system
    #exp_id = '{0}-seed-{1}'.format(args.exp_id, config.seed)
    exp_id = '{0}-seed-{1}'.format(args.exp_id, config_seed)    # Chris

    path_name = '{0}-shell-dist-{1}/agent_{2}'.format(name, exp_id, args.agent_id)
    log_dir = get_default_log_dir(path_name)
    logger = get_logger(log_dir=log_dir, file_name='train-log')
    config.logger = logger
    config.log_dir = log_dir

    # save shell config and env config
    #shutil.copy(shell_config_path, log_dir)
    try:
        with open(log_dir + '/shell_config.json', 'w') as f:
            json.dump(shell_config, f, indent=4)
            print('Shell configuration saved to shell_config.json')
    except:
        print('Something went wrong. Unable to save shell configuration json')
    shutil.copy(env_config_path, log_dir)

    # create/initialise agent
    logger.info('*****initialising agent {0}'.format(args.agent_id))



    num_tasks = len(set(shell_config['curriculum']['task_ids']))
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['curriculum']['task_ids']
    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        config.max_steps = [shell_config['curriculum']['max_steps'], ] * len(shell_config['curriculum']['task_ids'])

    #task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)          # Chris
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir)
    #eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    eval_task_fn= lambda log_dir: MetaCTgraphFlatObs(name, env_config_path,log_dir)            # Chris
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim,
        hidden_units=(200, 200, 200), num_tasks=num_tasks),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)

    config.seed = config_seed       # Chris

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    #for k, v in agent.network.named_parameters():       # Chris
    #    print(k, " : ", v)


    mask_interval = (config.max_steps[0]/(config.rollout_length * config.num_workers)) / 5

    # set up communication (transfer module)

    addresses = []
    ports = []
    file1 = open('./addresses.csv', 'r')
    lines = file1.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split(', ')
        if int(line[1]) == init_port and line[0] == init_address: continue
        addresses.append(line[0])
        ports.append(int(line[1]))

    manager = mp.Manager()
    knowledge_base = manager.dict()
    world_size = manager.Value('i', num_agents)

    mode = 'ondemand'
    comm = ParallelComm(agent.task_label_dim, agent.model_mask_dim, logger, init_address, init_port, mode, addresses, ports, knowledge_base, world_size)

    # start training
    shell_dist_train_mp(agent, comm, args.agent_id, args.num_agents, manager, knowledge_base, mask_interval)

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
    config.seed = 9157#config_seed              # Chris
    
    # set up logging system
    #exp_id = '{0}-seed-{1}'.format(args.exp_id, config.seed)
    exp_id = '{0}-seed-{1}'.format(args.exp_id, config_seed)    # Chris
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

    #task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)          # Chris
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir)
    #eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    eval_task_fn= lambda log_dir: MetaCTgraphFlatObs(name, env_config_path,log_dir)            # Chris
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim,
        hidden_units=(200, 200, 200), num_tasks=num_tasks),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)

    config.seed = config_seed       # Chris

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    #for k, v in agent.network.named_parameters():       # Chris
    #    print(k, " : ", v)

    addresses = []
    ports = []
    file1 = open('./addresses_eval.csv', 'r')
    lines = file1.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split(', ')
        if int(line[1]) == init_port and line[0] == init_address: continue
        addresses.append(line[0])
        ports.append(int(line[1]))

    manager = mp.Manager()
    knowledge_base = manager.dict()
    world_size = manager.Value('i', num_agents)

    # set up communication (transfer module)
    mode = 'ondemand'
    comm = ParallelCommEval(agent.task_label_dim, agent.model_mask_dim, logger, init_address, init_port, mode, addresses, ports, knowledge_base, world_size)

    # start training
    shell_dist_eval_mp(agent, comm, args.agent_id, args.num_agents, manager, knowledge_base)


##### Minigrid environment
def shell_dist_minigrid_mp(name, args, shell_config):
    shell_config_path = args.shell_config_path
    num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    init_address = args.ip#shell_config['dist_only']['init_address']
    init_port = args.port#shell_config['dist_only']['init_port']

    # set up config
    config = Config()
    config = global_config(config, name)
    # rescale state normaliser: suitable for grid encoding of states in minigrid
    config.state_normalizer = RescaleNormalizer(1./10.)

    # set seed
    config.seed = 9157#config_seed              # Chris
    
    # set up logging system
    #exp_id = '{0}-seed-{1}'.format(args.exp_id, config.seed)
    exp_id = '{0}-seed-{1}'.format(args.exp_id, config_seed)    # Chris

    path_name = '{0}-shell-dist-{1}/agent_{2}'.format(name, exp_id, args.agent_id)
    log_dir = get_default_log_dir(path_name)
    logger = get_logger(log_dir=log_dir, file_name='train-log')
    config.logger = logger
    config.log_dir = log_dir

    # save shell config and env config
    #shutil.copy(shell_config_path, log_dir)
    try:
        with open(log_dir + '/shell_config.json', 'w') as f:
            json.dump(shell_config, f, indent=4)
            print('Shell configuration saved to shell_config.json')
    except:
        print('Something went wrong. Unable to save shell configuration json')
    shutil.copy(env_config_path, log_dir)

    # create/initialise agent
    logger.info('*****initialising agent {0}'.format(args.agent_id))
    
    num_tasks = len(set(shell_config['curriculum']['task_ids']))
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['curriculum']['task_ids']
    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        config.max_steps = [shell_config['curriculum']['max_steps'], ] * len(shell_config['curriculum']['task_ids'])


    #task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, False)
    task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, 9157, False)          # Chris
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir, single_process=False)
    #eval_task_fn= lambda log_dir: MiniGridFlatObs(name, env_config_path,log_dir,config.seed,True)
    eval_task_fn= lambda log_dir: MiniGridFlatObs(name, env_config_path,log_dir, 9157, True)            # Chris
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim,
        hidden_units=(200, 200, 200), num_tasks=num_tasks),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)


    config.seed = config_seed       # Chris

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    #for k, v in agent.network.named_parameters():       # Chris
    #    print(k, " : ", v)


    # For optimal performance the mask interval should be divisible by the number of iterations per task
    # i..e, set max_steps to multiple of 512
    mask_interval = (config.max_steps[0]/(config.rollout_length * config.num_workers)) / 5

    addresses = []
    ports = []
    file1 = open('./addresses.csv', 'r')
    lines = file1.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split(', ')
        addresses.append(line[0])
        ports.append(int(line[1]))

    mode = 'ondemand'
    comm = ParallelComm(agent.task_label_dim, agent.model_mask_dim, logger, init_address, init_port, mode, mask_interval, addresses, ports)


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
    config.seed = 9157#config_seed
    
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

    #task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, False)
    task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, 9157, False)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir, single_process=False)
    #eval_task_fn= lambda log_dir: MiniGridFlatObs(name, env_config_path,log_dir,config.seed,True)
    eval_task_fn= lambda log_dir: MiniGridFlatObs(name, env_config_path,log_dir,9157,True)
    config.eval_task_fn = eval_task_fn
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim,
        hidden_units=(200, 200, 200), num_tasks=num_tasks),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)


    config.seed = config_seed       # Chris

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)
    
    #for k, v in agent.network.named_parameters():       # Chris
    #    print(k, " : ", v)

    addresses = []
    ports = []
    file1 = open('./addresses_eval.csv', 'r')
    lines = file1.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split(', ')
        addresses.append(line[0])
        ports.append(int(line[1]))

    mode = 'ondemand'
    comm = ParallelCommEval(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, init_address, init_port, mode, ports)


    # start evaluation on curriculum
    shell_dist_eval_mp(agent, comm, args.agent_id, args.num_agents)


##### ContinualWorld environment
# Need to add changes from Chris for seeds. Currently broken.
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


    mask_interval = (config.max_steps[0]/(config.rollout_length * config.num_workers)) / 5

    # set up communication (transfer module)
    mode = 'ondemand'
    comm = ParallelComm(args.agent_id, args.num_agents, agent.task_label_dim, \
        agent.model_mask_dim, logger, init_address, init_port, mode, mask_interval)

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
    parser.add_argument('agent_id', help='rank: the process id or machine id of the agent', type=int, default=0)        # Necessary for logging purposes and selecting the correcty curriculum from shell.json. Please set correctly.
    parser.add_argument('num_agents', help='world: total number of agents', type=int, default=1)                        # Required for the moment.
    parser.add_argument('--shell_config_path', help='shell config', default='./shell_4x4.json')                             # Change the default so you don't have to type it every time. Required.
    parser.add_argument('--exp_id', help='id of the experiment. useful for setting '\
        'up structured directory of experiment results/data', default='upz', type=str)                                  # Not required
    parser.add_argument('--mode', help='indicate evaluation agent', type=int, default=0)                                # Don't change. Default value of 0 is the only one that is implemented and tested at the moment.
    parser.add_argument('--port', help='port to use for this agent', type=int, default=29500)                           # Required. Port for the listening server.
    parser.add_argument('--ip', help='ip address to use for this agent', type=str, default='127.0.0.1')                 # Required. IP for the listening server. Default is localhost.
    parser.add_argument('--shuffle', help='randomise the task curriculum', type=int, default=0)                         # Not required. If you want to randomise the order of tasks in the curriculum then you can change to 1
    args = parser.parse_args()

    print(args)

    with open(args.shell_config_path, 'r') as f:
        shell_config = json.load(f)
        shell_config['curriculum'] = shell_config['agents'][args.agent_id]

        # Randomise the curriculum
        if args.shuffle == 1 and args.mode != 1:
            random.shuffle(shell_config['curriculum']['task_ids'])

        shell_config['seed'] = shell_config['seed'][args.agent_id]      # Chris
        del shell_config['agents'][args.agent_id]

    if shell_config['env']['env_name'] == 'minigrid':
        name = Config.ENV_MINIGRID
        if args.mode == 1:
            shell_dist_minigrid_eval(name, args, shell_config)
        elif args.mode == 0:
            shell_dist_minigrid_mp(name, args, shell_config)
        else:
            raise ValueError('--mode {0} not implemented'.format(args.mode))

    elif shell_config['env']['env_name'] == 'ctgraph':
        name = Config.ENV_METACTGRAPH
        if args.mode == 1:
            shell_dist_mctgraph_eval(name, args, shell_config)
        elif args.mode == 0:
            shell_dist_mctgraph_mp(name, args, shell_config)
        else:
            raise ValueError('--mode {0} not implemented'.format(args.mode))

    elif shell_config['env']['env_name'] == 'continualworld':
        name = Config.ENV_CONTINUALWORLD
        if args.mode == 1:
            shell_dist_continualworld_eval(name, args, shell_config)
        elif args.mode == 0:
            shell_dist_continualworld_mp(name, args, shell_config)
        else:
            raise ValueError('--mode {0} not implemented'.format(args.mode))

    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))
