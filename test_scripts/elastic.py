import multiprocess as mp
import torch.distributed as dist
from torch.distributed.elastic.agent.server import WorkerSpec
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
import random
import time
import argparse
import torch
from deep_rl import *


def trainer(comm):
    # Pseudo trainer

    # Set initial task
    msg = random.choice([np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])])
    print('TASK CHANGE: ', msg)
    iteration = 0

    # Start iteration training
    while True:
        # Do communication
        other_agents_request = comm.send_receive_request(msg)
        print(other_agents_request)
        msg = None

        # Increment iteration
        iteration += 1
        time.sleep(1)

        # Check if its time to switch task
        if iteration == 512:
            msg = random.choice([np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])])
            print('TASK CHANGE: ', msg)
            iteration = 0






if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    mp.set_start_method('fork', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id', help='rank: the process id or machine id of the agent', type=int)
    parser.add_argument('num_agents', help='world: total number of agents', type=int)
    parser.add_argument('--shell_config_path', help='shell config', default='./shell.json')
    parser.add_argument('--exp_id', help='id of the experiment. useful for setting '\
        'up structured directory of experiment results/data', default='upz', type=str)
    args = parser.parse_args()

    print(args)

    with open(args.shell_config_path, 'r') as f:
        shell_config = json.load(f)
        shell_config['curriculum'] = shell_config['agents'][args.agent_id]
        del shell_config['agents'][args.agent_id]

    if shell_config['env']['env_name'] == 'minigrid':
        name = Config.ENV_MINIGRID


        shell_dist_minigrid_mp(name, args, shell_config)


    elif shell_config['env']['env_name'] == 'ctgraph':
        name = Config.ENV_METACTGRAPH
        shell_dist_mctgraph(name, args, shell_config)
    elif shell_config['env']['env_name'] == 'continualworld':
        name = Config.ENV_CONTINUALWORLD
        shell_dist_continualworld(name, args, shell_config)
    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))