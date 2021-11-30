#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
single task learning experiments (standard deepRL)
'''

import json
import copy
import shutil
import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

########## dynamic_grid
def dqn_dynamic_grid(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.seed = 4185
    random_seed(config.seed)
    exp_id = '-unique_exp_id_here'
    log_name = name + '-dqn' + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.task_fn = lambda log_dir: DynamicGridFlatObs(name, log_dir=log_dir)
    config.eval_task_fn = config.task_fn
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr, alpha=0.95, eps=0.01)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim,
        body=FCBody(state_dim, hidden_units=(512,1024,1024,32)))
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 7e4))
    config.replay_fn = lambda: Replay(memory_size=int(7e4), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = RescaleNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 5000
    config.exploration_steps= 15000
    config.logger = get_logger(log_dir=config.log_dir)
    config.double_q = False
    config.sgd_update_frequency = 7
    config.max_steps = int(1.12e5) + 1 # approx 112,000 steps
    config.episode_log_interval = 100
    agent = DQNAgent(config)
    config.agent_name = agent.__class__.__name__
    task = agent.task.random_tasks(num_tasks=1, requires_task_label=False)[0]
    agent.task.reset_task(task)
    states = agent.task.reset_task(task)
    agent.states = agent.config.state_normalizer(states)
    with open('{0}/task_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(task, f)

	# run experiment
    run_episodes(agent)

    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)


def a2c_dynamic_grid(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.seed = 4185
    random_seed(config.seed)
    exp_id = '-unique_exp_id_here'
    log_name = name + '-a2c' + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 16
    task_fn = lambda log_dir: DynamicGridFlatObs(name, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim,
        phi_body=FCBody(state_dim, hidden_units=(200, 200, 200)), 
        actor_body=DummyBody(200), 
        critic_body=DummyBody(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 7
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(1.12e5) + 1 # approx 112,000 steps
    config.logger = get_logger(log_dir=config.log_dir)
    agent = A2CAgent(config)
    config.agent_name = agent.__class__.__name__
    task = agent.task.random_tasks(num_tasks=1, requires_task_label=False)[0]
    agent.task.reset_task(task)
    states = agent.task.reset_task(task)
    agent.states = agent.config.state_normalizer(states)
    with open('{0}/task_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(task, f)

	# run experiment
    run_iterations(agent)

    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

def ppo_dynamic_grid(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.seed = 8379
    random_seed(config.seed)
    exp_id = '-unique_exp_id_here'
    log_name = name + '-ppo' + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 16
    task_fn = lambda log_dir: DynamicGridFlatObs(name, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim,
        phi_body=FCBody(state_dim, hidden_units=(200, 200, 200)), 
        actor_body=DummyBody(200), 
        critic_body=DummyBody(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 7
    config.optimization_epochs = 4
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(1.12e5) + 1 # approx 112,000 steps
    config.logger = get_logger(log_dir=config.log_dir)
    agent = PPOAgent(config)
    config.agent_name = agent.__class__.__name__
    task = agent.task.random_tasks(num_tasks=1, requires_task_label=False)[0]
    states = agent.task.reset_task(task)
    agent.states = agent.config.state_normalizer(states)
    with open('{0}/task_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(task, f)

	# run experiment
    run_iterations(agent)

    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)


########## ctgraph
## dqn
def dqn_ctgraph(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.seed = 4185
    random_seed(config.seed)
    exp_id = '-unique_exp_id_here'
    log_name = name + '-dqn' + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.task_fn = lambda log_dir: CTgraphFlatObs(name, env_config_path=env_config_path, log_dir=log_dir)
    config.eval_task_fn = config.task_fn
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr, alpha=0.95, eps=0.01)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim,
        body=FCBody(state_dim, hidden_units=(512,1024,1024,32)))
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 7e4))
    config.replay_fn = lambda: Replay(memory_size=int(7e4), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = RescaleNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 5000
    config.exploration_steps= 15000
    config.logger = get_logger(log_dir=config.log_dir)
    config.double_q = False
    config.sgd_update_frequency = 7
    config.max_steps = int(1.12e5) + 1 # approx 112,000 steps
    config.episode_log_interval = 100
    agent = DQNAgent(config)
    config.agent_name = agent.__class__.__name__
    task = agent.task.random_tasks(num_tasks=1, requires_task_label=False)[0]
    agent.task.reset_task(task)
    states = agent.task.reset_task(task)
    agent.states = agent.config.state_normalizer(states)
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/task_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(task, f)

	# run experiment
    run_episodes(agent)

    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

def a2c_ctgraph(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.seed = 4185
    random_seed(config.seed)
    exp_id = '-unique_exp_id_here'
    log_name = name + '-a2c' + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 16
    assert env_config_path is not None, '`env_config_path` should be set for the CTgraph environnent'
    task_fn = lambda log_dir: CTgraphFlatObs(name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim,
        phi_body=FCBody(state_dim, hidden_units=(200, 200, 200)), 
        actor_body=DummyBody(200), 
        critic_body=DummyBody(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 7
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(1.12e5) + 1 # approx 112,000 steps
    config.logger = get_logger(log_dir=config.log_dir)
    agent = A2CAgent(config)
    config.agent_name = agent.__class__.__name__
    task = agent.task.random_tasks(num_tasks=1, requires_task_label=False)[0]
    agent.task.reset_task(task)
    states = agent.task.reset_task(task)
    agent.states = agent.config.state_normalizer(states)
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/task_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(task, f)

	# run experiment
    run_iterations(agent)

    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

## ppo
def ppo_ctgraph(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.seed = 8379
    random_seed(config.seed)
    exp_id = '-unique_exp_id_here'
    log_name = name + '-ppo' + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 16
    assert env_config_path is not None, '`env_config_path` should be set for the CTgraph environnent'
    task_fn = lambda log_dir: CTgraphFlatObs(name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim,
        phi_body=FCBody(state_dim, hidden_units=(200, 200, 200)), 
        actor_body=DummyBody(200), 
        critic_body=DummyBody(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 7
    config.optimization_epochs = 4
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(1.12e5) + 1 # approx 112,000 steps
    config.logger = get_logger(log_dir=config.log_dir)
    agent = PPOAgent(config)
    config.agent_name = agent.__class__.__name__
    task = agent.task.random_tasks(num_tasks=1, requires_task_label=False)[0]
    states = agent.task.reset_task(task)
    agent.states = agent.config.state_normalizer(states)
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/task_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(task, f)

	# run experiment
    run_iterations(agent)

    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    random_seed(42)
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    # dynamic-grid experiments
    #game = 'DynamicGrid-v0'
    #ppo_dynamic_grid(name=game)
    #a2c_dynamic_grid(name=game)
    #dqn_dynamic_grid(name=game)

    # ctgraph experiments
    game = 'CTgraph-v0'
    env_config_path = './ctgraph.json'
    ppo_ctgraph(name=game, env_config_path=env_config_path)
    #a2c_ctgraph(name=game, env_config_path=env_config_path)
    #dqn_ctgraph(name=game, env_config_path=env_config_path)
