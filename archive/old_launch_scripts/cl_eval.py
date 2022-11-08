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
    'A2CAgentSCP': A2CAgentSCP,
    'A2CAgentEWC': A2CAgentEWC,
    'A2CAgentBaseline': A2CAgentBaseline,
    'DQNAgentBaseline': DQNAgentBaseline
}

## ctgraph
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
    if config.agent_name == 'DQNAgentBaseline':
        config.network_fn = lambda state_dim, action_dim, label_dim: VanillaNet_CL(action_dim, \
            label_dim, body=FCBody_CL(state_dim, task_label_dim=label_dim, \
            hidden_units=(512,1024,1024,32)))
        config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 7e4))
        config.replay_fn = lambda: Replay(memory_size=int(7e4), batch_size=32)
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

