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
os.environ["CUDA_VISIBLE_DEVICES"]="0"


## ppo
def ppo_ctgraph_cl_l1_weights(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'baseline' # or 'mas' or 'ewc' or 'baseline'
    config.seed = 8379
    random_seed(config.seed)
    exp_id = '-l1-weights'
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 16
    assert env_config_path is not None, '`env_config_path` should be set for the CTgraph environnent'
    task_fn = lambda log_dir: CTgraphFlatObs(name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
        state_dim, action_dim, label_dim, 
        phi_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200)), 
        actor_body=DummyBody(200), 
        critic_body=DummyBody(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.75
    config.rollout_length = 7
    config.optimization_epochs = 4
    config.num_mini_batches = 4
    config.cl_num_learn_blocks = 1
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(5.6e4)+1 # note, max steps per task
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir)
    config.cl_num_learn_blocks = 1
    config.cl_num_tasks = 4
    config.cl_requires_task_label = True
    config.cl_alpha = 0.25
    config.cl_loss_coeff = 0.5 # for scp
    config.cl_n_slices = 200
    config.reg_loss_coeff = 5e-4
    if config.cl_preservation == 'mas': agent = PPOAgentMAS(config)
    elif config.cl_preservation == 'scp': agent = PPOAgentSCP(config)
    elif config.cl_preservation == 'ewc': agent = PPOAgentEWC(config)
    #elif config.cl_preservation == 'baseline': agent = PPOAgentBaseline(config)
    elif config.cl_preservation == 'baseline': agent = PPOAgentBaselineL1Weights(config)
    else: raise ValueError('config.cl_preservation should be set to \'mas\' or \'scp\' or \'ewc\'.')
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    tasks = [tasks[0], tasks[3]] # NOTE
    config.cl_num_tasks = 2 # NOTE
    config.cl_num_learn_blocks = 3 # NOTE
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

def ppo_ctgraph_cl_l2_weights(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'baseline' # or 'mas' or 'ewc' or 'baseline'
    config.seed = 8379
    random_seed(config.seed)
    exp_id = '-l2-weights'
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 16
    assert env_config_path is not None, '`env_config_path` should be set for the CTgraph environnent'
    task_fn = lambda log_dir: CTgraphFlatObs(name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
        state_dim, action_dim, label_dim, 
        phi_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200)), 
        actor_body=DummyBody(200), 
        critic_body=DummyBody(200))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.75
    config.rollout_length = 7
    config.optimization_epochs = 4
    config.num_mini_batches = 4
    config.cl_num_learn_blocks = 1
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(5.6e4)+1 # note, max steps per task
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir)
    config.cl_num_learn_blocks = 1
    config.cl_num_tasks = 4
    config.cl_requires_task_label = True
    config.cl_alpha = 0.25
    config.cl_loss_coeff = 0.5 # for scp
    config.cl_n_slices = 200
    config.reg_loss_coeff = 5e-4
    if config.cl_preservation == 'mas': agent = PPOAgentMAS(config)
    elif config.cl_preservation == 'scp': agent = PPOAgentSCP(config)
    elif config.cl_preservation == 'ewc': agent = PPOAgentEWC(config)
    #elif config.cl_preservation == 'baseline': agent = PPOAgentBaseline(config)
    elif config.cl_preservation == 'baseline': agent = PPOAgentBaselineL2Weights(config)
    else: raise ValueError('config.cl_preservation should be set to \'mas\' or \'scp\' or \'ewc\'.')
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    tasks = [tasks[0], tasks[3]] # NOTE
    config.cl_num_tasks = 2 # NOTE
    config.cl_num_learn_blocks = 3 # NOTE
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
    #random_seed(42)
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    # ctgraph experiments
    game = 'CTgraph-v0'
    env_config_path = './ctgraph.json'
    #ppo_ctgraph_cl_l1_weights(name=game, env_config_path=env_config_path)
    ppo_ctgraph_cl_l2_weights(name=game, env_config_path=env_config_path)
