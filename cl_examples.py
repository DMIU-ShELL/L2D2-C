#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
continual learning experiments with weight preservation (consolidation) in RL
'''

import shutil
import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"


## dynamic-grid
def a2c_dynamic_grid_cl(name, env_config_path=None):
    config = Config()
    config.log_dir = get_default_log_dir(name + '-a2c')
    config.num_workers = 1
    task_fn = lambda log_dir: DynamicGrid(name, max_steps=200, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, MNISTConvBody())
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(1e5) # note, max steps per task
    config.logger = get_logger(log_dir=config.log_dir)
    agent = A2CAgent(config)
    config.cl_num_tasks = 3
    config.cl_requires_task_label = True
    tasks = agent.task.get_all_tasks(config.cl_requires_task_label)
    tasks = tasks[ : config.cl_num_tasks]
    # save tasks
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_cl(agent, cl_config)

def ppo_dynamic_grid_cl(name, env_config_path=None):
    raise NotImplementedError

## ctgraph
def a2c_ctgraph_1dstates_cl(name, env_config_path=None):
    # states are 1d feature vectors in this ctgraph configuration.
    config = Config()
    config.log_dir = get_default_log_dir(name + '-a2c')
    config.num_workers = 1
    assert env_config_path is not None, '`env_config_path` should be set for the CTgraph environnent'
    task_fn = lambda log_dir: CTgraph(name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
        state_dim, action_dim, label_dim, FCBody_CL(state_dim, label_dim, hidden_units=(64, 16)))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.iteration_log_interval = 100
    #config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = int(5e4) # note, max steps per task
    config.logger = get_logger(log_dir=config.log_dir)
    agent = A2CContinualLearnerAgent(config)
    config.cl_num_tasks = 3
    config.cl_requires_task_label = True
    tasks = agent.task.get_all_tasks(config.cl_requires_task_label)
    tasks = tasks[ : config.cl_num_tasks]
    # save env_config and tasks
    shutil.copy(env_config_path, config.log_dir)
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_cl(agent, tasks)

def a2c_ctgraph_cl(name, env_config_path=None):
    config = Config()
    config.log_dir = get_default_log_dir(name + '-a2c')
    config.num_workers = 1
    assert env_config_path is not None, '`env_config_path` should be set for the CTgraph environnent'
    task_fn = lambda log_dir: CTgraph(name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, CTgraphConvBody())
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(1e5) # note, max steps per task
    config.logger = get_logger(log_dir=config.log_dir)
    agent = A2CContinualLearnerAgent(config)
    config.cl_num_tasks = 3
    config.cl_requires_task_label = True
    tasks = agent.task.get_all_tasks(config.cl_requires_task_label)
    tasks = tasks[ : config.cl_num_tasks]
    # save env_config and tasks
    shutil.copy(env_config_path, config.log_dir)
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_cl(agent, tasks)

def ppo_ctgraph_cl(name, env_config_path=None):
    raise NotImplementedError

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    random_seed(42)
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    # dynamic-grid experiments
    game = 'DynamicGrid-v0'
    # a2c_dynamic_grid_cl(name=game)
    # ppo_dynamic_grid_cl(name=game)

    # ctgraph experiments
    game = 'CTgraph-v0'
    env_config_path = './ctgraph.json'
    a2c_ctgraph_1dstates_cl(name=game, env_config_path=env_config_path)
    # a2c_ctgraph_cl(name=game, env_config_path=env_config_path)
    # ppo_ctgraph_cl(name=game, env_config_path=env_config_path)
