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
os.environ["CUDA_VISIBLE_DEVICES"]="3"

########## dynamic_grid
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
    raise NotImplementedError

########## ctgraph
## dqn
def dqn_ctgraph_1dstates_cl(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    #config.cl_preservation = 'baseline'
    #config.cl_preservation = 'mas'
    config.cl_preservation = 'ewc'
    #config.cl_preservation = 'scp'
    config.seed = 4185
    random_seed(config.seed)
    #exp_id = '-wo_pres'
    exp_id = '-pres_ac'
    log_name = name + '-dqn' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.task_fn = lambda log_dir: CTgraph(name, env_config_path=env_config_path, log_dir=log_dir)
    config.eval_task_fn = config.task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr, alpha=0.95, eps=0.01)
    config.network_fn = lambda state_dim, action_dim, label_dim: VanillaNet_CL(action_dim, label_dim,
        body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(512,1024,1024,32)))
    #config.network_fn =lambda state_dim, action_dim, label_dim: DuelingNet_CL(action_dim, label_dim\
    #    body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(512,1024,1024,32)))
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 7e4))
    config.replay_fn = lambda: Replay(memory_size=int(7e4), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    #config.reward_normalizer = SignNormalizer()
    config.reward_normalizer = RescaleNormalizer()
    config.discount = 0.99
    #config.target_network_update_freq = 10000
    config.target_network_update_freq = 5000
    #config.exploration_steps= 50000
    config.exploration_steps= 15000
    config.logger = get_logger(log_dir=config.log_dir)
    config.double_q = False
    config.sgd_update_frequency = 7
    config.max_steps = 1e5 # note, max steps per task
    config.episode_log_interval = 100
    config.evaluation_episodes = 10

    config.cl_num_tasks = 4
    config.cl_requires_task_label = True
    config.cl_alpha = 0.25
    #config.cl_loss_coeff = 1e8 # for ewc
    #config.cl_loss_coeff = 0.5 # for mas
    config.cl_loss_coeff = 0.5 # for scp
    if config.cl_preservation == 'mas': agent = DQNAgentMAS(config)
    elif config.cl_preservation == 'scp': agent = DQNAgentSCP(config)
    elif config.cl_preservation == 'ewc': agent = DQNAgentEWC(config)
    elif config.cl_preservation == 'baseline': agent = DQNAgentBaseline(config)
    else: raise ValueError('config.cl_preservation should be set to \'mas\' or \'scp\' or \'ewc\'.')
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_episodes_cl(agent, tasks)

    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

def a2c_ctgraph_1dstates_cl(name, env_config_path=None):
    # states are 1d feature vectors in this ctgraph configuration.
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    #config.lr = 0.0007
    config.lr = 0.00015
    config.cl_preservation = 'baseline'
    #config.cl_preservation = 'mas'
    #config.cl_preservation = 'ewc'
    config.seed = 4185
    random_seed(config.seed)
    exp_id = '-wo_pres'
    #exp_id = '-pres_ac_neuromod'
    log_name = name + '-a2c' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 8
    assert env_config_path is not None, '`env_config_path` should be set for the CTgraph environnent'
    task_fn = lambda log_dir: CTgraph(name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.eval_task_fn = task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL_NM(
        state_dim, action_dim, label_dim, 
        phi_body=None,
        actor_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(512,1024,1024,32)), 
        critic_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(512,1024,1024,32)))
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 7
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(1e5) # note, max steps per task
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir)

    config.cl_num_tasks = 4
    config.cl_requires_task_label = True
    config.cl_alpha = 0.25
    #config.cl_loss_coeff = 1e8 # for ewc
    config.cl_loss_coeff = 2e8 # for ewc with nm
    #config.cl_loss_coeff = 1e1 # for mas
    if config.cl_preservation == 'mas': agent = A2CAgentMAS(config)
    elif config.cl_preservation == 'scp': agent = A2CAgentSCP(config)
    elif config.cl_preservation == 'ewc': agent = A2CAgentEWC(config)
    elif config.cl_preservation == 'baseline': agent = A2CAgentBaseline(config)
    else: raise ValueError('config.cl_preservation should be set to \'mas\' or \'scp\' or \'ewc\'.')
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
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

## ppo
def ppo_ctgraph_cl(name, env_config_path=None):
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'scp' # or 'mas' or 'ewc' or 'baseline'
    config.seed = 8379
    random_seed(config.seed)
    exp_id = ''
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
    #config.entropy_weight = 0.01
    config.entropy_weight = 0.75
    config.rollout_length = 7
    config.optimization_epochs = 4
    config.num_mini_batches = 4
    config.cl_num_learn_blocks = 1
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 100
    config.gradient_clip = 5
    config.max_steps = int(5e4) # note, max steps per task
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir)
    config.cl_num_learn_blocks = 1
    config.cl_num_tasks = 4
    config.cl_requires_task_label = True
    config.cl_alpha = 0.25
    config.cl_loss_coeff = 0.5 # for scp
    config.cl_n_slices = 200
    if config.cl_preservation == 'mas': agent = PPOAgentMAS(config)
    elif config.cl_preservation == 'scp': agent = PPOAgentSCP(config)
    elif config.cl_preservation == 'ewc': agent = PPOAgentEWC(config)
    elif config.cl_preservation == 'baseline': agent = PPOAgentBaseline(config)
    else: raise ValueError('config.cl_preservation should be set to \'mas\' or \'scp\' or \'ewc\'.')
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    tasks = [tasks[0], tasks[3], tasks[0], tasks[3], tasks[0], tasks[3], tasks[0], tasks[3], tasks[0], tasks[3], tasks[0], tasks[3]] # NOTE
    config.cl_num_tasks = 12
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

    # dynamic-grid experiments
    #game = 'DynamicGrid-v0'
    # a2c_dynamic_grid_cl(name=game)

    # ctgraph experiments
    game = 'CTgraph-v0'
    env_config_path = './ctgraph.json'
    #dqn_ctgraph_1dstates_cl(name=game, env_config_path=env_config_path)
    #a2c_ctgraph_1dstates_cl(name=game, env_config_path=env_config_path)
    ppo_ctgraph_cl(name=game, env_config_path=env_config_path)
