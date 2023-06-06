import json
import shutil
import matplotlib
matplotlib.use("Pdf")
import multiprocessing as mp
from deep_rl.utils.misc import mkdir, get_default_log_dir, run_episodes
from deep_rl.utils.torch_utils import set_one_thread, random_seed, select_device
from deep_rl.utils.config import Config
from deep_rl.component.policy import SamplePolicy
from deep_rl.component.replay import Replay
from deep_rl.component.random_process import GaussianProcess
from deep_rl.utils.normalizer import ImageNormalizer, RescaleNormalizer, RunningStatsNormalizer, RewardRunningStatsNormalizer
from deep_rl.utils.logger import get_logger
from deep_rl.utils.trainer_shell import shell_dist_train_mp, shell_dist_eval_mp
from deep_rl.utils.trainer_ll import run_iterations_w_oracle
from deep_rl.utils.schedule import LinearSchedule
from deep_rl.agent.PPO_agent import ShellAgent_DP
from deep_rl.agent.TD3_agent import TD3Agent
from deep_rl.shell_modules.communication.comms import ParallelComm, ParallelCommEval, ParallelCommOmniscient
from deep_rl.component.task import ParallelizedTask, MiniGridFlatObs, MetaCTgraphFlatObs, ContinualWorld
from deep_rl.network.network_heads import TD3Net, CategoricalActorCriticNet_SS, GaussianActorCriticNet_SS
from deep_rl.network.network_bodies import FCBody, FCBody_SS, DummyBody_CL
import argparse
import torch
import random
import torch.nn.functional as F



def global_config(config, name):
    # ctgraph config
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
    config.rollout_length = 320
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = 25600
    config.evaluation_episodes = 25
    config.cl_requires_task_label = True
    config.task_fn = None
    config.eval_task_fn = None
    config.network_fn = None 
    config.eval_interval = None#1
    return config


def td3_baseline_mctgraph(name, args, shell_config):
    env_config_path = shell_config['env']['env_config_path']
    config_seed = shell_config['seed']
    init_port = args.port

    config = Config()
    config = global_config(config, name)
    config.state_normalizer = RescaleNormalizer()

    config.seed = 9157

    exp_id = '{0}-seed-{1}'.format(args.exp_id, config_seed)

    path_name = '{0}-shell-dist-{1}/agent_{2}'.format(name, exp_id, args.agent_id)
    log_dir = get_default_log_dir(path_name)
    logger = get_logger(log_dir=log_dir, file_name='train-log')
    config.logger = logger
    config.log_dir = log_dir

    try:
        with open(log_dir + '/shell_config.json', 'w') as f:
            json.dump(shell_config, f, indent=4)
            print('Shell configuration saved to shell_config.json')
    except:
        print('Something went terribly wrong. Unable to save shell configuration JSON')
    shutil.copy(env_config_path, log_dir)

    logger.info('*****initialising agent {0}'.format(args.agent_id))

    num_tasks = len(set(shell_config['curriculum']['task_ids']))
    config.cl_num_tasks = num_tasks
    config.task_ids = shell_config['curriculum']['task_ids']
    if isinstance(shell_config['curriculum']['max_steps'], list):
        config.max_steps = shell_config['curriculum']['max_steps']
    else:
        config.max_steps = [shell_config['curriculum']['max_steps'], ] * len(shell_config['curriculum']['task_ids'])

    task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn,config.num_workers,log_dir=config.log_dir, single_process=False)
    
    eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(name, env_config_path, log_dir)
    config.eval_task_fn = eval_task_fn

    config.network_fn = lambda state_dim, action_dim: TD3Net(
        action_dim,
        actor_body_fn=lambda: FCBody(state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            state_dim+action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))
    
    config.seed = config_seed
    config.use_task_label = False

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda action_dim: GaussianProcess(
        size=(action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    #config.warm_up = int(1e4)
    config.min_memory_size = int(1e4)
    config.target_network_mix = 1e-3

    agent = TD3Agent(config)
    run_episodes(agent)



if __name__ == '__main__':
    mkdir('log')
    set_one_thread()

    mp.set_start_method('fork', force=True)


    parser = argparse.ArgumentParser()
    #parser.add_argument('--algo', help='algorithm to run')
    parser.add_argument('agent_id', help='rank: the process id or machine id of the agent', type=int)
    parser.add_argument('port', help='port to use for this agent', type=int)
    parser.add_argument('--shell_config_path', help='path to environment config', default='./shell_1x1.json')
    parser.add_argument('--exp_id', help='id of the experiment. useful for setting up structured directory of experiment results/data', default='upz', type=str)
    args = parser.parse_args()

    with open(args.shell_config_path, 'r') as f:
        # Load shell configuration JSON
        shell_config = json.load(f)
        shell_config['curriculum'] = shell_config['agents'][args.agent_id]

        # Randomise the curriculum if shuffle raised and not in evaluation mode
        #if args.shuffle and not args.eval: random.shuffle(shell_config['curriculum']['task_ids'])

        # Handle seeds
        shell_config['seed'] = shell_config['seed'][args.agent_id]      # Chris
        del shell_config['agents'][args.agent_id]

    if shell_config['env']['env_name'] == 'ctgraph':
        name = Config.ENV_METACTGRAPH
        td3_baseline_mctgraph(name, args, shell_config)
