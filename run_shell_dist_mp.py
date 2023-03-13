#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


#  _______                .__                        .__                             .__                       .___             
#  \      \  __ __   ____ |  |   ____ _____ _______  |  | _____   __ __  ____   ____ |  |__     ____  ____   __| _/____   ______
#  /   |   \|  |  \_/ ___\|  | _/ __ \\__  \\_  __ \ |  | \__  \ |  |  \/    \_/ ___\|  |  \  _/ ___\/  _ \ / __ |/ __ \ /  ___/
# /    |    \  |  /\  \___|  |_\  ___/ / __ \|  | \/ |  |__/ __ \|  |  /   |  \  \___|   Y  \ \  \__(  <_> ) /_/ \  ___/ \___ \ 
# \____|__  /____/  \___  >____/\___  >____  /__|    |____(____  /____/|___|  /\___  >___|  /  \___  >____/\____ |\___  >____  >
#         \/            \/          \/     \/                  \/           \/     \/     \/       \/           \/    \/     \/ 




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
from deep_rl.shell_modules.communication.comms import ParallelComm, ParallelCommEval, ParallelCommOmniscient
from deep_rl.component.task import ParallelizedTask, MiniGridFlatObs, MetaCTgraphFlatObs, ContinualWorld
from deep_rl.network.network_heads import CategoricalActorCriticNet_SS, GaussianActorCriticNet_SS
from deep_rl.network.network_bodies import FCBody_SS, DummyBody_CL
import argparse
import torch
import random
import time



# helper function
def global_config(config, name):
    '''# ctgraph config
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
    config.entropy_weight = 0.00015 #0.75
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = 25600
    config.evaluation_episodes = 5
    config.cl_requires_task_label = True
    config.task_fn = None
    config.eval_task_fn = None
    config.network_fn = None 
    config.eval_interval = None#1
    return config'''

    # minigrid config
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
    config.evaluation_episodes = 5#50
    config.cl_requires_task_label = True
    config.task_fn = None
    config.eval_task_fn = None
    config.network_fn = None 
    config.eval_interval = None
    return config

'''
ShELL distributed system with multiprocessing.
Currently only communication. Data collection and model optimisation to be added
in the future.
'''
##### (Meta)CT-graph
def shell_dist_mctgraph_mp(name, args, shell_config):
    # this might be deprecated
    #shell_config_path = args.shell_config_path
    #num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    init_port = args.port

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
        print('Something went terribly wrong. Unable to save shell configuration JSON')
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
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir, single_process=False)
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
    config.use_task_label = False   # Chris

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    # Uncomment to print out the network parameters for visualisation. Can be useful for debugging.
    #for k, v in agent.network.named_parameters():       # Chris
    #    print(k, " : ", v)

    querying_frequency = (config.max_steps[0]/(config.rollout_length * config.num_workers)) / args.comm_interval

    addresses, ports = [], []
    reference_file = open(args.reference, 'r')
    lines = reference_file.readlines()
    
    if args.quick_start:
        for i in range(args.agent_id + 1):
            line = lines[i].strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))

    else:
        for line in lines:
            line = line.strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))
        

    # Initialize dictionary to store the most up-to-date rewards for a particular embedding/task label.
    manager = mp.Manager()
    knowledge_base = manager.dict()
    mode = manager.Value('b', args.omni)

    # If True then run the omnisicent mode agent, otherwise run the normal agent.
    if mode.value:
        comm = ParallelCommOmniscient(args.num_agents, agent.task_label_dim, agent.model_mask_dim, logger, init_port, zip(addresses, ports), knowledge_base, manager, args.localhost, mode, args.dropout)
        shell_dist_train_mp(agent, comm, args.agent_id, args.num_agents, manager, knowledge_base, querying_frequency, mode, args.amnesia)
    
    
    comm = ParallelComm(args.num_agents, agent.task_label_dim, agent.model_mask_dim, logger, init_port, zip(addresses, ports), knowledge_base, manager, args.localhost, mode, args.dropout)

    # start training
    shell_dist_train_mp(agent, comm, args.agent_id, args.num_agents, manager, knowledge_base, querying_frequency, mode, args.amnesia)

def shell_dist_mctgraph_eval(name, args, shell_config):
    shell_config_path = args.shell_config_path
    num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    init_port = args.port

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
    #shutil.copy(shell_config_path, log_dir)
    try:
        with open(log_dir + '/shell_config.json', 'w') as f:
            json.dump(shell_config, f, indent=4)
            print('Shell configuration saved to shell_config.json')
    except:
        print('Something went terribly wrong. Unable to save shell configuration JSON')
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
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir, single_process=False)
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
    config.use_task_label = False   # Chris

    agent = ShellAgent_DP(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.agent_id)

    #for k, v in agent.network.named_parameters():       # Chris
    #    print(k, " : ", v)

    addresses, ports = [], []
    reference_file = open(args.reference, 'r')
    lines = reference_file.readlines()
    
    if args.quick_start:
        for i in range(args.agent_id + 1):
            line = lines[i].strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))

    else:
        for line in lines:
            line = line.strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))

    manager = mp.Manager()
    knowledge_base = manager.dict()

    # set up communication (transfer module)
    mode = 'ondemand'
    comm = ParallelCommEval(args.num_agents, agent.task_label_dim, agent.model_mask_dim, logger, init_port, zip(addresses, ports), knowledge_base, manager, args.localhost)

    # start training
    shell_dist_eval_mp(agent, comm, args.agent_id, args.num_agents, manager, knowledge_base)


##### Minigrid environment
def shell_dist_minigrid_mp(name, args, shell_config):
    #shell_config_path = args.shell_config_path
    #num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    #init_address = args.ip#shell_config['dist_only']['init_address']
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

    # Uncomment to print out the network parameters for visualisation. Can be useful for debugging.
    #for k, v in agent.network.named_parameters():       # Chris
    #    print(k, " : ", v)

    querying_frequency = (config.max_steps[0]/(config.rollout_length * config.num_workers)) / args.comm_interval

    addresses, ports = [], []
    reference_file = open(args.reference, 'r')
    lines = reference_file.readlines()
    
    if args.quick_start:
        for i in range(args.agent_id + 1):
            line = lines[i].strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))

    else:
        for line in lines:
            line = line.strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))
        

    # Initialize dictionary to store the most up-to-date rewards for a particular embedding/task label.
    manager = mp.Manager()
    knowledge_base = manager.dict()
    mode = manager.Value('b', args.omni)

    # If True then run the omnisicent mode agent, otherwise run the normal agent.
    if mode.value:
        comm = ParallelCommOmniscient(args.num_agents, agent.task_label_dim, agent.model_mask_dim, logger, init_port, zip(addresses, ports), knowledge_base, manager, args.localhost, mode, args.dropout)
        shell_dist_train_mp(agent, comm, args.agent_id, args.num_agents, manager, knowledge_base, querying_frequency, mode, args.amnesia)
    
    
    comm = ParallelComm(args.num_agents, agent.task_label_dim, agent.model_mask_dim, logger, init_port, zip(addresses, ports), knowledge_base, manager, args.localhost, mode, args.dropout)

    # start training
    shell_dist_train_mp(agent, comm, args.agent_id, args.num_agents, manager, knowledge_base, querying_frequency, mode, args.amnesia)

def shell_dist_minigrid_eval(name, args, shell_config):
    shell_config_path = args.shell_config_path
    #num_agents = args.num_agents

    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    # address and port number of the master/first agent (rank/id 0) in the pool of agents
    #init_address = shell_config['dist_only']['init_address']
    init_port = args.port#shell_config['dist_only']['init_port']

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
    #shutil.copy(shell_config_path, log_dir)
    try:
        with open(log_dir + '/shell_config.json', 'w') as f:
            json.dump(shell_config, f, indent=4)
            print('Shell configuration saved to shell_config.json')
    except:
        print('Something went terribly wrong. Unable to save shell configuration JSON')
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

    addresses, ports = [], []
    reference_file = open(args.reference, 'r')
    lines = reference_file.readlines()
    
    if args.quick_start:
        for i in range(args.agent_id + 1):
            line = lines[i].strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))

    else:
        for line in lines:
            line = line.strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))

    manager = mp.Manager()
    knowledge_base = manager.dict()

    # set up communication (transfer module)
    mode = 'ondemand'
    comm = ParallelCommEval(args.num_agents, agent.task_label_dim, agent.model_mask_dim, logger, init_port, zip(addresses, ports), knowledge_base, manager, args.localhost)

    # start training
    shell_dist_eval_mp(agent, comm, args.agent_id, args.num_agents, manager, knowledge_base)


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


    querying_frequency = (config.max_steps[0]/(config.rollout_length * config.num_workers)) / args.comm_interval

    addresses, ports = [], []
    reference_file = open(args.reference, 'r')
    lines = reference_file.readlines()
    
    if args.quick_start:
        for i in range(args.agent_id + 1):
            line = lines[i].strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))

    else:
        for line in lines:
            line = line.strip('\n').split(', ')
            addresses.append(line[0])
            ports.append(int(line[1]))
        

    # Initialize dictionary to store the most up-to-date rewards for a particular embedding/task label.
    manager = mp.Manager()
    knowledge_base = manager.dict()

    mode = 'ondemand'
    comm = ParallelComm(args.num_agents, agent.task_label_dim, agent.model_mask_dim, logger, init_port, mode, zip(addresses, ports), knowledge_base, manager, args.localhost)

    # start training
    shell_dist_train_mp(agent, comm, args.agent_id, args.num_agents, manager, knowledge_base, querying_frequency)

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

    time.sleep(0.5)
    mkdir('log')
    set_one_thread()

    mp.set_start_method('fork', force=True) # Set multiprocessing method as fork. Only available on UNIX (i.e., MacOS and Linux). DMIU is not currently compatible with Windows.

    ##################################################################################################################################################################################################################
    #                                                                                            DMIU LAUNCH ARGUMENTS                                                                                               #
    ##################################################################################################################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id', help='rank: the process id or machine id of the agent', type=int)                   # NOTE: REQUIRED Used to create the logging filepath and select a specific curriculum from the shell configuration JSON.
    parser.add_argument('port', help='port to use for this agent', type=int)                                            # NOTE: REQUIRED Port for the listening server.
    parser.add_argument('--num_agents', help='world: total number of agents', type=int, default=1)                      # Will eventually be deprecated. Currently used to set the communication module initial world size.
    parser.add_argument('--shell_config_path', help='shell config', default='./shell_minigrid10.json')                       # File path to your chosen shell.json configuration file. Changing the default here might save you some time.
    parser.add_argument('--exp_id', help='id of the experiment. useful for setting '\
        'up structured directory of experiment results/data', default='upz', type=str)                                  # Experiment ID. Can be useful for setting up directories for logging results/data.
    parser.add_argument('--eval', '--e', '-e', help='launches agent in evaluation mode', action='store_true')           # Flag used to start the system in evaluation agent mode. By default the system will run in learning mode.
    parser.add_argument('--omni', '--o', '-o', help='launches agetn in omniscient mode. omniscient agents use the '\
        'gather all querying method to gather all knowledge from the network while still operating as a functional '\
            'learning agent', action='store_true')                                                                      # Flag used to start the system in omniscient agent mode. By default the system will run in learning mode.
                                                                                                                        # Omnisicient agent mode cannot be combined with evaluation mode.

    parser.add_argument('--localhost', '--ls', '-ls', help='used to run DMIU in localhost mode', action='store_true')   # Flag used to start the system using localhost instead of public IP. Can be useful for debugging network related problems.
    parser.add_argument('--shuffle', '--s', '-s', help='randomise the task curriculum', action='store_true')            # Not required. If you want to randomise the order of tasks in the curriculum then you can change to 1
    parser.add_argument('--comm_interval', '--i', '-i', help='integer value indicating the number of communications '\
        'to perform per task', type= int, default=5)                                                                    # Configures the communication interval used to test and take advantage of the lucky agent phenomenon. We found that a value of 5 works well. 
                                                                                                                        # Please do not modify this value unless you know what you're doing as it may cause unexpected results.

    parser.add_argument('--quick_start', '--qs', '-qs', help='use this to take advantage of the quick start method ' \
        'for the reference table', action='store_true')                                                                 # Quick start allows you to use a complete reference table across all your agents, to quickly start localised experimentation. To use this
                                                                                                                        # you will need to simply put every address of all agents you intend to run on your system and use the same file on all your agents at launch.
                                                                                                                        # Then you must use the unique values for your agents using the agent_id argument and start each agent sequentially. I.e., for 16 agents you 
                                                                                                                        # should have agents with ids 0 ~ 15. This will make each agent populate its known peers lists with all agents that have been started up to that point.
                                                                                                                        # Was implemented just to make internal testing easier during development.

    parser.add_argument('--device', help='select device 1 for GPU or 0 for CPU. default is GPU', type=int, default=1)   # Used to select device. By default system will try to use the GPU. Currently PyTorch is only compatible with NVIDIA GPUs or Apple M Series processors.
    parser.add_argument('--reference', '--r', '-r', help='reference.csv file path', type=str, default='reference.csv')
    parser.add_argument('--dropout', '--d', '-d', help='Communication dropout parameter', type=float, default=0.0)
    parser.add_argument('--amnesia', '--a', '-a', help='probability of total memory loss', type=float, default = 0.0)
    args = parser.parse_args()

    select_device(args.device)
    print(args)

    with open(args.shell_config_path, 'r') as f:
        # Load shell configuration JSON
        shell_config = json.load(f)
        shell_config['curriculum'] = shell_config['agents'][args.agent_id]

        # Randomise the curriculum if shuffle raised and not in evaluation mode
        if args.shuffle and not args.eval: random.shuffle(shell_config['curriculum']['task_ids'])

        # Handle seeds
        shell_config['seed'] = shell_config['seed'][args.agent_id]      # Chris
        del shell_config['agents'][args.agent_id]


    # Parse arguments and launch the correct environment-agent configuration.
    if shell_config['env']['env_name'] == 'minigrid':
        name = Config.ENV_MINIGRID
        if args.eval:
            shell_dist_minigrid_eval(name, args, shell_config)

        else:
            shell_dist_minigrid_mp(name, args, shell_config)

    elif shell_config['env']['env_name'] == 'ctgraph':
        name = Config.ENV_METACTGRAPH
        if args.eval:
            shell_dist_mctgraph_eval(name, args, shell_config)

        else:
            shell_dist_mctgraph_mp(name, args, shell_config)

    elif shell_config['env']['env_name'] == 'continualworld':
        name = Config.ENV_CONTINUALWORLD
        if args.eval:
            shell_dist_continualworld_eval(name, args, shell_config)

        else:
            shell_dist_continualworld_mp(name, args, shell_config)

    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))
