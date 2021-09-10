#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
continual learning evaluations of trained models in RL
'''

import json
import shutil
import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

mapper = {
    'A2CAgentMAS': A2CAgentMAS,
    'A2CAgentSCP': A2CAgentSCP
}

## ctgraph
def a2c_ctgraph_1dstates_cl(name, env_config_path=None):
    # states are 1d feature vectors in this ctgraph configuration.
    config = Config()
    config.cl_preservation = 'mas'
    config.seed = 1125
    random_seed(config.seed)
    config.log_dir = get_default_log_dir(name + '-a2c' + '-' + config.cl_preservation + '-extra-small-actor')
    config.num_workers = 4
    assert env_config_path is not None, '`env_config_path` should be set for the CTgraph environnent'
    task_fn = lambda log_dir: CTgraph(name, env_config_path=env_config_path, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    config.evaluation_env = CTgraph(name, env_config_path=env_config_path, log_dir=config.log_dir)
    #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025)
    #config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
    #    state_dim, action_dim, label_dim, FCBody_CL(state_dim, label_dim, hidden_units=(64, 128, 16)))
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
        state_dim, action_dim, label_dim, 
        phi_body=None,
        #actor_body=FCBody_CL(state_dim=state_dim, task_label_dim=label_dim, hidden_units=(128, 1024, 512, 32)), 
        actor_body=FCBody_CL(state_dim=state_dim, task_label_dim=label_dim, hidden_units=(128, 64, 32)), 
        critic_body=FCBody_CL(state_dim=state_dim, task_label_dim=label_dim, hidden_units=(1024, 1024, 512, 32)) )
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.01
    config.rollout_length = 7
    config.iteration_log_interval = 100
    #config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = int(1e5) # note, max steps per task
    config.logger = get_logger(log_dir=config.log_dir)

    config.cl_num_tasks = 3
    config.cl_requires_task_label = True
    config.cl_alpha = 0.5
    #config.cl_loss_coeff = 1e6
    config.cl_loss_coeff = 5e5
    if config.cl_preservation == 'mas':
        input('mas (press enter):')
        agent = A2CAgentMAS(config)
    elif config.cl_preservation == 'scp':
        input('scp (press enter):')
        agent = A2CAgentSCP(config)
    else:
        raise ValueError('config.cl_preservation should be set to \'mas\' or \'scp\'.')
    tasks = agent.task.get_all_tasks(config.cl_requires_task_label)
    tasks = tasks[ : config.cl_num_tasks]
    config.cl_num_tasks = 6
    import copy
    tasks = tasks + copy.deepcopy(tasks)
    # save env_config and tasks
    shutil.copy(env_config_path, config.log_dir)
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_cl(agent, tasks)

def eval_ctgraph(args, config):
    config.task_fn = None
    config.env_config_path = args.exp_path + '/env_config.json'
    config.eval_task_fn = lambda log_dir: CTgraph(config.env_name, config.env_config_path, 
        config.log_dir)
    config.state_normalizer = ImageNormalizer()
    # NOTE, hack optimizer only added to make code run in eval. Otherwise, it is not used in eval.
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL(
        state_dim, action_dim, label_dim, 
        phi_body=None,
        actor_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(512,1024,1024,32)), 
        critic_body=FCBody_CL(state_dim, task_label_dim=label_dim, hidden_units=(512,1024,1024,32)))
    agent = mapper[config.agent_name](config) # load agent
    agent.load('{0}/{1}-{2}-model-{3}.bin'.format(args.exp_path, config.agent_name, config.tag, 
        config.env_name)) # load model
    # load tasks
    with open('{0}/tasks_info.bin'.format(args.exp_path), 'rb') as f: tasks = pickle.load(f)
    run_evals_cl(agent, tasks, args.num_evals)
    return

def eval_dynamic_grid(args, config):
    return

def eval_agent(args):
    config = Config()
    # load config
    with open('{0}/config.json'.format(args.exp_path), 'r') as f:
        config.merge(config_dict=json.load(f))
    config.task_fn = None # disable env instance for training. only keep env for eval instance.
    if args.seed is not None: config.seed = args.seed
    random_seed(config.seed)

    # setup eval logs
    config.log_dir = '{0}/eval_logs/{1}'.format(args.exp_path, get_time_str())
    config.logger = get_logger(log_dir=config.log_dir)

    if config.env_name == 'CTgraph-v0': eval_ctgraph(args, config)
    elif config.env_name == 'DynamicGrid-v0': eval_dynamic_grid(args, config)
    else: raise NotImplementedError
    return

if __name__ == '__main__':
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', help='path to experiment logs', type=str)
    parser.add_argument('num_evals', help='number of evalulation episodes', type=int)
    parser.add_argument('--seed', help='seed for evaluation run', type=int, default=None)
    eval_agent(parser.parse_args())

