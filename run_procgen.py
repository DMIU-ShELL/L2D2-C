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

import json
import shutil
import matplotlib
matplotlib.use("Pdf")
import multiprocessing as mp
from deep_rl.utils.misc import mkdir, get_default_log_dir
from deep_rl.utils.torch_utils import set_one_thread, random_seed, select_device
from deep_rl.utils.config import Config
from deep_rl.utils.normalizer import ImageNormalizer, RescaleNormalizer, RunningStatsNormalizer, RewardRunningStatsNormalizer
from deep_rl.utils.logger import get_logger
from deep_rl.utils.trainer_shell import trainer_learner, trainer_evaluator
from deep_rl.component.policy import SamplePolicy
from deep_rl.component.task import ParallelizedTask, Procgen
from deep_rl.network.network_heads import CategoricalActorCriticNet_SS, GaussianActorCriticNet_SS, CategoricalActorCriticNet_SS_Comp
from deep_rl.network.network_bodies import FCBody_SS, DummyBody_CL, ConvBody_SS
from deep_rl.agent.PPO_agent import PPODetectShell, PPOShellAgent
from deep_rl.agent.SAC_agent import SACDetectShell, SACShellAgent

from deep_rl.shell_modules.communication.comms import ParallelComm, ParallelCommEval, ParallelCommOmniscient
from deep_rl.shell_modules.communication.comms_detect import ParallelCommDetect, ParallelCommDetectEval
from deep_rl.shell_modules.detect.detect import Detect

import argparse
import torch
import random

# helper functions
def global_config(config, name):
    config.env_name = name
    config.env_config_path = None
    config.lr = 0.00015
    config.cl_preservation = 'supermask'
    config.seed = None
    config.backbone_seed = 9157
    config.log_dir = None
    config.logger = None 
    config.num_workers = 1
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)

    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01 #0.75
    config.rollout_length = 2048
    config.optimization_epochs = 8
    config.num_mini_batches = 32
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    config.gradient_clip = 40
    config.max_steps = 25600
    config.evaluation_episodes = 5#50
    config.cl_requires_task_label = True
    config.task_fn = None
    config.eval_task_fn = None
    config.network_fn = None 
    config.eval_interval = None
    return config

def setup_configs_and_logs(config, args, shell_config, global_config):
    config = global_config(config, name)

    env_config_path = shell_config['env']['env_config_path']
    config.seed = shell_config['seed']
    random_seed(config.seed)
    config.backbone_seed = 9157
    config.init_port = args.port

    ###############################################################################
    # Detect Module
    config.detect_reference_num = shell_config['detect_reference_num']
    config.detect_num_samples = shell_config['detect_num_samples']
    config.emb_dist_threshold = shell_config['emb_dist_threshold']
    config.detect_module_activation_frequency = shell_config['detect_module_activation_frequency']


    ###############################################################################
    # Logging
    exp_id = '{0}-seed-{1}'.format(args.exp_id, config.seed)
    path_name = args.pathheader + '/{0}-shell-dist-{1}/agent_{2}'.format(name, exp_id, args.curriculum_id)
    log_dir = get_default_log_dir(path_name)
    logger = get_logger(log_dir=log_dir, file_name='train-log')
    config.logger = logger
    config.log_dir = log_dir

    # save shell config and env config
    #shutil.copy(shell_config_path, log_dir)
    with open(log_dir + '/shell_config.json', 'w') as f:
        json.dump(shell_config, f, indent=4)
    shutil.copy(env_config_path, log_dir)

    # create/initialise agent
    logger.info('*****initialising L2D2-C agent')


    print(shell_config)

    ###############################################################################
    # Curriculum setup. TODO: Check how much of this is needed. Remove unnecessary stuff
    if 'task_ids' in shell_config['curriculum']:
        num_tasks = len(set(shell_config['curriculum']['task_ids']))
    else:
        num_tasks = len(shell_config['curriculum']['task_paths'])

    config.cl_num_tasks = num_tasks
    config.task_paths, config.task_ids = None, None
    if 'task_paths' in shell_config['curriculum']:
        config.task_paths = shell_config['curriculum']['task_paths']
    elif 'task_ids' in shell_config['curriculum']:
        config.task_ids = shell_config['curriculum']['task_ids']
    else:
        raise Exception('shell config is missing task_ids or task_paths')

    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        if 'task_paths' in shell_config['curriculum']:
            config.max_steps = [shell_config['curriculum']['max_steps'], ] * len(shell_config['curriculum']['task_paths'])
        elif 'task_ids' in shell_config['curriculum']:
            config.max_steps = [shell_config['curriculum']['max_steps'], ] * len(shell_config['curriculum']['task_ids'])

    return config, env_config_path

def detect_finalise_and_run(config, Agent):
    config.use_task_label = False #Chris    # Saptarshi: What is this for?


    ###############################################################################
    # Setup detect module
    #Passing the Detect Module in the config object of the Agent OPTIONAL COULD BE USED BY THE TRAINER ONLY
    config.detect_fn = lambda reference_num, input_dim, num_samples: Detect(reference_num, input_dim, num_samples, one_hot=True, normalized=True)


    ###############################################################################
    # Initialise Manager() for handling shared variables over an internal server
    config.manager = mp.Manager()
    config.seen_tasks = config.manager.dict()
    config.mode = config.manager.Value('b', args.omni)
    config.evaluator_present = config.manager.Value('b', False)


    ###############################################################################
    # Setup agent module
    #if args.eval:
    #    config.entropy_weight = 0
    agent = Agent(config)
    config.agent_name = agent.__class__.__name__ + '_{0}'.format(args.curriculum_id)

    # Communication frequency. TODO: This will need a rework if we don't know the length of task encounters.
    config.querying_frequency = 10#(config.max_steps[0]/(config.rollout_length * config.num_workers)) / args.comm_interval


    ###############################################################################
    # Read the reference ip-port pairs to enter a collective. Setup the parallelised
    # communication module.
    addresses, ports = [], []
    reference_file = open(args.reference, 'r')
    lines = reference_file.readlines()
    for line in lines:
        line = line.strip('\n').split(', ')
        addresses.append(line[0])
        ports.append(int(line[1]))
        
    # If True then run the omnisicent mode agent, otherwise run the traditional agent.
    # TODO: Have to figure out how a traditional learner can transition to omniscient whenever required. Not really implemented yet and doesn't really work properly.
    #if GLOBAL_mode.value:
    #    comm = ParallelCommOmniscient(agent.get_task_emb_size(), agent.model_mask_dim, config, zip(addresses, ports), GLOBAL_task_record, GLOBAL_manager, args.localhost, GLOBAL_mode, args.dropout, config.emb_dist_threshold) #Chris added threshold
    #    trainer_learner(agent, comm, args.curriculum_id, GLOBAL_manager, GLOBAL_task_record, config.querying_frequency, GLOBAL_mode)


    # Comm hyperparameters
    config.query_wait = 0.2 # ms
    config.mask_wait = 0.3  # ms
    config.top_k = 5 # get top 5 masks for collective linear comb
    config.reward_progression_factor = 0.6 # x * self.current_task_reward < sender_rw @send_mask_requests() # NOTE: NOT USED ANYMORE
    config.reward_stability_threshold = 0.6 # Reward threshold at which point we don't want the agent to query anymore for stability

    # Log all system hyperparameters and settings to log directory
    config.log_hyperparameters(config.logger.log_dir + '/parameters.txt')

    if args.eval:
        comm = ParallelCommDetectEval(
            task_label_dim=agent.task_label_dim, 
            mask_dim=agent.model_mask_dim, 
            logger=config.logger, 
            init_port=config.init_port, 
            reference=zip(addresses, ports), 
            knowledge_base=config.seen_tasks, 
            manager=config.manager, 
            localhost=args.localhost, 
            mode=config.mode, 
            dropout=args.dropout, 
            threshold=config.emb_dist_threshold
        )
        # Start training
        trainer_evaluator(agent, comm, args.curriculum_id, config.manager, config.seen_tasks)

    else:
        comm = ParallelCommDetect(
            embd_dim = agent.get_task_emb_size(), 
            mask_dim = agent.model_mask_dim, 
            reference = zip(addresses, ports), 
            args = args,
            config = config
        )
        # Start evaluating
        trainer_learner(agent, comm, args.curriculum_id, config.manager, config.querying_frequency, config.mode)
         

'''
Lifelong Learning Distributed and Decentralised (L2D2-C) experiments
Multi-agent continual lifelong learners

Developed as part of work supported by the Defense Advanced Research Projects Agency
(DARPA) under contract no. HR00112190132 (Shared Experience Lifelong Learning).

Each agent is based on ppo and the modulating masks
lifelong reinforcement learning algorithm.
https://arxiv.org/abs/2212.11110
'''

# main experiment methods. currently implemented: meta-ctgraph with ppo. TODO: Implement evaluation agents with task labels instead of embeddings
def procgen_ppo(name, args, shell_config):
    # Initialise config object
    config = Config()
    config, env_config_path = setup_configs_and_logs(config, args, shell_config, global_config)


    ###############################################################################
    # ENVIRONMENT SPECIFIC SETUP. SETUP TRAINING AND EVALUATION TASK FUNCTIONS
    # AND THE NETWORK FUNCTION.
    # Training task lambda function
    task_fn = lambda log_dir: Procgen(name=name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir, single_process=True)

    # Evaluation task mabda function. TODO: Is the evaluation task function necessary for a traditional learner?
    eval_task_fn = lambda log_dir: Procgen(name=name, env_config_path=env_config_path, log_dir=log_dir)
    config.eval_task_fn = eval_task_fn

    # Network lambda function
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS_Comp(\
        state_dim, action_dim, label_dim,
        phi_body=ConvBody_SS(
            state_dim, 
            feature_dim=512,
            task_label_dim=label_dim,
            num_tasks=config.cl_num_tasks, 
            new_task_mask=args.new_task_mask,
            seed=config.seed
            ),
        actor_body=DummyBody_CL(512),
        critic_body=DummyBody_CL(512),
        num_tasks=config.cl_num_tasks,
        new_task_mask=args.new_task_mask,
        seed=config.seed)    # 'random' for mask RI. 'linear_comb' for mask LC.
    
    # Environment sepcific setup ends.
    ###############################################################################
    

    # Select what agent to use here. Default is *DetectShell which is an Modulating Masks PPO agent that uses the
    # Wasserstein detect module for online task identity inference.
    detect_finalise_and_run(config, PPODetectShell)


# main experiment methods. currently implemented: meta-ctgraph with ppo. TODO: Implement evaluation agents with task labels instead of embeddings
def procgen_ppo_eval(name, args, shell_config):
    # Initialise config object
    config = Config()
    config, env_config_path = setup_configs_and_logs(config, args, shell_config, global_config)


    ###############################################################################
    # ENVIRONMENT SPECIFIC SETUP. SETUP TRAINING AND EVALUATION TASK FUNCTIONS
    # AND THE NETWORK FUNCTION.
    # Training task lambda function
    task_fn = lambda log_dir: Procgen(name=name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir, single_process=True)

    # Evaluation task mabda function. TODO: Is the evaluation task function necessary for a traditional learner?
    eval_task_fn= lambda log_dir: Procgen(name=name, env_config_path=env_config_path, log_dir=log_dir)
    config.eval_task_fn = eval_task_fn

    # Network lambda function
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS_Comp(\
        state_dim, action_dim, label_dim,
        phi_body=ConvBody_SS(
            state_dim, 
            feature_dim=512,
            task_label_dim=label_dim,
            num_tasks=config.cl_num_tasks, 
            new_task_mask=args.new_task_mask,
            seed=config.seed
            ),
        actor_body=DummyBody_CL(512),
        critic_body=DummyBody_CL(512),
        num_tasks=config.cl_num_tasks,
        new_task_mask=args.new_task_mask,
        seed=config.seed)    # 'random' for mask RI. 'linear_comb' for mask LC.
    
    # Environment sepcific setup ends.
    ###############################################################################
    

    # Select what agent to use here. Default is *DetectShell which is an Modulating Masks PPO agent that uses the
    # Wasserstein detect module for online task identity inference.
    detect_finalise_and_run(config, PPOShellAgent)



if __name__ == '__main__':
    mkdir('log')
    set_one_thread()

    mp.set_start_method('fork', force=True) # Set multiprocessing method as fork. Only available on UNIX (i.e., MacOS and Linux). DMIU is not currently compatible with Windows.

    ##################################################################################################################################################################################################################
    #                                                                                            LAUNCH ARGUMENTS                                                                                                    #
    ##################################################################################################################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('curriculum_id', help='index of the curriculum to use from the shell config json', type=int)                   # NOTE: REQUIRED Used to create the logging filepath and select a specific curriculum from the shell configuration JSON.
    parser.add_argument('port', help='port to use for this agent', type=int)                                            # NOTE: REQUIRED Port for the listening server.
    parser.add_argument('--shell_config_path', help='shell config', default='./shell_configs/the_chosen_one/procgen_multi.json')                         # File path to your chosen shell.json configuration file. Changing the default here might save you some time.
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
        'to perform per task', type= int, default=20)                                                                    # Configures the communication interval used to test and take advantage of the lucky agent phenomenon. We found that a value of 5 works well. 
                                                                                                                        # Please do not modify this value unless you know what you're doing as it may cause unexpected results.

    parser.add_argument('--device', help='select device 1 for GPU or 0 for CPU. default is GPU', type=int, default=1)   # Used to select device. By default system will try to use the GPU. Currently PyTorch is only compatible with NVIDIA GPUs or Apple M Series processors.
    parser.add_argument('--reference', '--r', '-r', help='reference.csv file path', type=str, default='reference.csv')
    parser.add_argument('--dropout', '--d', '-d', help='Comunication dropout parameter', type=float, default=0.0)
    parser.add_argument('--new_task_mask', help='', default='linear_comb', type=str)
    parser.add_argument('--pathheader', '--p', '-p', help='experiment header to log path for launcher.py', type=str, default='')
    args = parser.parse_args()

    select_device(args.device)
    print(args)

    with open(args.shell_config_path, 'r') as f:
        # Load shell configuration JSON
        shell_config = json.load(f)
        shell_config['curriculum'] = shell_config['agents'][args.curriculum_id]

        # Randomise the curriculum if shuffle raised and not in evaluation mode
        if args.shuffle and not args.eval: random.shuffle(shell_config['curriculum']['task_ids'])

        # Handle seeds
        shell_config['seed'] = shell_config['seed'][args.curriculum_id]      # Chris
        shell_config['detect_reference_num'] = shell_config['detect_reference_num']#[args.agent_id]#Chris

        shell_config['detect_num_samples'] = shell_config['detect_num_samples']#Chris
        
        shell_config['emb_dist_threshold'] = shell_config['emb_dist_threshold']#[args.agent_id]# Chris

        shell_config['detect_module_activation_frequency'] = shell_config['detect_module_activation_frequency']#[args.agent_id] #Chris
        
        del shell_config['agents'][args.curriculum_id]

    if args.pathheader == '':
        args.pathheader = shell_config['env']['env_name']
        
    # Parse arguments and launch the correct environment-agent configuration.
    if shell_config['env']['env_name'] == 'procgen':
        name = Config.ENV_PROCGEN
        print(name)
        if args.eval:
            procgen_ppo(name, args, shell_config)

        else:
            procgen_ppo(name, args, shell_config)

    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))
