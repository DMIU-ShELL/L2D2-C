import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import dynamic_grid

def experiment():
    # MAKE SURE TO RUN THESE ON THE LAB MACHINES TO ENSURE THE PARALLELIZATION CAN HAPPEN. I think we can change the configuration to
    # Match how we need the environment to change. Can do the configuration before the loop, then within the loop access the environment
    # Using the Classical Control thing and then change the environment as needed.

    config = Config()
    # initializing the configurations
    name = 'DynamicGrid-v0'
    #name = 'CartPole-v0'
    # Getting the name of the environment we want to work out of
    config.log_dir = get_default_log_dir(name + '-A2C')
    config.num_workers = 1
    task_fn = lambda log_dir: DynamicGrid(name, max_steps=200, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=config.log_dir, single_process=True)


    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.00025)
    #config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        #state_dim, action_dim, FCBody(state_dim))
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, MNISTConvBody())
    config.policy_fn = SamplePolicy
    config.discount = 0.99
    config.logger = get_logger(log_dir=config.log_dir)
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.max_steps = 3e5
    config.iteration_log_interval = 100
    # Setting the configurations as we desire them to be
    run_iterations_experiment(A2CAgent(config))




if __name__ == '__main__':
    import torch
    mkdir('log')
    #mkdir('tf_log')
    set_one_thread()
    random_seed(42)

    # -1 is CPU, a positive integer is the index of GPU
    select_device(0)

    config.num_workers = 0
    # task_fn = lambda log_dir: ClassicalControl(name, max_steps=200, log_dir=log_dir)
    #
    # config.task_fn = lambda: BaseTask(task_fn, config.num_workers,
    #                                           log_dir=get_default_log_dir(experiment.__name__))
    # # Switched this to base task so that I coud try running it on my machine
    #
    # config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    # config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
    #     state_dim, action_dim, FCBody(state_dim))
    # config.policy_fn = SamplePolicy
    # config.discount = 0.99
    # config.logger = get_logger()
    # config.gae_tau = 1.0
    # config.entropy_weight = 0.01
    # config.rollout_length = 5
    # print("I got here")
    # print(config.task_fn)
    experiment()
