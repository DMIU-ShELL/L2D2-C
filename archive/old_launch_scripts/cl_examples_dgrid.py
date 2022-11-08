#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
continual learning experiments with weight preservation (consolidation) in RL
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


## ppo
def ppo_dgrid_cl(name, env_config_path=None): # no sparsity, no consolidation (pure baseline)
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'baseline'
    config.seed = 8379
    random_seed(config.seed)
    exp_id = ''
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4
    task_fn = lambda log_dir: DynamicGridFlatObs(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
        state_dim, action_dim, label_dim, 
        phi_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200)), 
        actor_body=DummyBody_CL(200), 
        critic_body=DummyBody_CL(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1
    config.rollout_length = 8
    config.optimization_epochs = 4
    config.num_mini_batches = 16
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 10
    config.gradient_clip = 5
    config.max_steps = 5e4
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir)
    config.cl_requires_task_label = True
    config.cl_num_tasks = 8
    agent = PPOAgentBaseline(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    if env_config_path is not None:
        shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_cl(agent, tasks)
    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

def ppo_scp_dgrid_cl(name, env_config_path=None): # no sparsity, scp consolidation
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'scp'
    config.seed = 8379
    random_seed(config.seed)
    exp_id = ''
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4
    task_fn = lambda log_dir: DynamicGridFlatObs(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
        state_dim, action_dim, label_dim, 
        phi_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200)), 
        actor_body=DummyBody_CL(200), 
        critic_body=DummyBody_CL(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1
    config.rollout_length = 8
    config.optimization_epochs = 4
    config.num_mini_batches = 16
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 10
    config.gradient_clip = 5
    config.max_steps = 5e4
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir)
    config.cl_requires_task_label = True
    config.cl_num_tasks = 8

    config.cl_alpha = 0.25
    config.cl_loss_coeff = 0.5 # for scp
    config.cl_n_slices = 200

    agent = PPOAgentSCP(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    if env_config_path is not None:
        shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_cl(agent, tasks)
    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

## neurmodulated masking of forward pass. neuromodulated network generates masks per layer. the masks
## are then applied to the weights of the target/base network during forward pass.
def ppo_dgrid_cl_nm_mask_fp(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'scp'
    config.seed = 8379
    random_seed(config.seed)
    exp_id = '-nm-mask-fp'
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4
    task_fn = lambda log_dir: DynamicGridFlatObs(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL_Mask(
        state_dim, action_dim, label_dim, 
        phi_body=FCBody_CL_Mask(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200)), 
        actor_body=DummyBody_CL_Mask(200), 
        critic_body=DummyBody_CL_Mask(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1
    config.rollout_length = 8
    config.optimization_epochs = 4
    config.num_mini_batches = 16
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 10
    config.gradient_clip = 5
    config.max_steps = 5e4
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir)
    config.cl_requires_task_label = True
    config.cl_num_tasks = 8

    config.cl_alpha = 0.25
    config.cl_loss_coeff = 0.5 # for scp
    config.cl_n_slices = 200

    agent = PPOAgentSCPModulatedFP(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    if env_config_path is not None:
        shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_cl(agent, tasks)
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
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    # dynamic grid experiments
    game = 'DynamicGrid-v0'
    env_config_path = None

    parser = argparse.ArgumentParser()
    parser.add_argument('algo', help='algorithm to run')
    args = parser.parse_args()

    if args.algo == 'baseline':
        ppo_dgrid_cl(name=game, env_config_path=env_config_path)
    elif args.algo == 'baseline_scp':
        ppo_scp_dgrid_cl(name=game, env_config_path=env_config_path)
    elif args.algo == 'nm_mask_fp':
        ppo_dgrid_cl_nm_mask_fp(name=game, env_config_path=env_config_path)
    else:
        raise ValueError('not implemented')
