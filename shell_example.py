#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
Shared Experience Lifelong Learning (ShELL) experiments
Multi-agent continual lifelong learners
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

def shell_minigrid(name, env_config_path=None): # ppo with supermask superposition
    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'ss'
    config.seed = 8379
    random_seed(config.seed)
    exp_id = ''
    log_name = name + '-shell' + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4
    assert env_config_path is not None,'`env_config_path` should be set for the MiniGrid environment'

    with open(shell_config_path, 'r'_ as f:
        shell_config = json.load(f)

    task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, False)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, True)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(
        state_dim, action_dim, label_dim, 
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200), num_tasks=num_tasks), 
        actor_body=DummyBody_CL(200), 
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks)
    config.policy_fn = SamplePolicy
    #config.state_normalizer = ImageNormalizer()
    config.state_normalizer = RescaleNormalizer(1./10.)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1 #0.75
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 10
    config.gradient_clip = 5
    config.max_steps = 1e6
    config.evaluation_episodes = 100
    config.logger = get_logger(log_dir=config.log_dir)
    config.cl_requires_task_label = True

    agents = []
    num_agents = len(shell_config['config'])
    for idx in range(num_agents):
        env_config_path = shell_config['config'][idx]['env_config_path']
        config_ = copy.deepcopy(config)
        task_fn= lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config_.seed, False)
        config_.task_fn=lambda: ParallelizedTask(task_fn,config_.num_workers,log_dir=config_.log_dir)
        eval_task_fn=lambda log_dir: MiniGridFlatObs(name, env_config_path,log_dir,config_.seed,True)
        config_.eval_task_fn = eval_task_fn
        config.max_steps = 5e4
        config.evaluation_episodes = 10

        # get num_tasks from env_config
        with open(env_config_path, 'r') as f:
            env_config_ = json.load(f)
        num_tasks = len(env_config_['tasks'])
        del env_config_
        config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(\
            state_dim, action_dim, label_dim, 
            phi_body=FCBody_SS(state_dim, task_label_dim=label_dim, \
            hidden_units=(200, 200, 200), num_tasks=num_tasks), 
            actor_body=DummyBody_CL(200), 
            critic_body=DummyBody_CL(200),
            num_tasks=num_tasks)

        agent = PPOAgentSS(config_)
        config_.agent_name = agent.__class__.__name__ + '_{0}'.format(idx)
        agents.append(agent)

    #tasks = agent.config.cl_tasks_info
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    run_iterations_ss(agents)
    ## save config
    #with open('{0}/config.json'.format(config.log_dir), 'w') as f:
    #    dict_config = vars(config)
    #    for k in dict_config.keys():
    #        if not isinstance(dict_config[k], int) \
    #        and not isinstance(dict_config[k], float) and dict_config[k] is not None:
    #            dict_config[k] = str(dict_config[k])
    #    json.dump(dict_config, f)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    # minigrid experiments
    game = 'MiniGrid'

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_config_path', help='environment config', default=None)
    args = parser.parse_args()

    shell_minigrid(name=game, env_config_path=args.env_config_path)
