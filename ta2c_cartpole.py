#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## cart pole

def dqn_cart_pole():
    game = 'CartPole-v0'
    config = Config()
    config.task_fn = lambda: ClassicalControl(game, max_steps=200,log_dir=get_default_log_dir(dqn_cart_pole.__name__))
    config.evaluation_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, FCBody(state_dim))
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, FCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e4))
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    config.logger = get_logger()
    config.double_q = True
    # config.double_q = False
    run_episodes(DQNAgent(config))

def a2c_cart_pole():
    config = Config()
    name = 'CartPole-v0'
    # name = 'MountainCar-v0'
    task_fn = lambda log_dir: ClassicalControl(name, max_steps=200, log_dir=log_dir)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(a2c_cart_pole.__name__))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, FCBody(state_dim))
    config.policy_fn = SamplePolicy
    config.discount = 0.99
    config.logger = get_logger()
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    run_iterations(A2CAgent(config))

def categorical_dqn_cart_pole():
    game = 'CartPole-v0'
    config = Config()
    config.task_fn = lambda: ClassicalControl(game, max_steps=200)
    config.evaluation_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: \
        CategoricalNet(action_dim, config.categorical_n_atoms, FCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e4))
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.logger = get_logger(skip=True)
    config.categorical_v_max = 100
    config.categorical_v_min = -100
    config.categorical_n_atoms = 50
    run_episodes(CategoricalDQNAgent(config))

def quantile_regression_dqn_cart_pole():
    config = Config()
    config.task_fn = lambda: ClassicalControl('CartPole-v0', max_steps=200)
    config.evaluation_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, FCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(0.1))
    config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.logger = get_logger(skip=True)
    config.num_quantiles = 20
    run_episodes(QuantileRegressionDQNAgent(config))

def n_step_dqn_cart_pole():
    config = Config()
    task_fn = lambda log_dir: ClassicalControl('CartPole-v0', max_steps=200, log_dir=log_dir)
    config.evaluation_env = task_fn(None)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, FCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e4))
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.logger = get_logger()
    run_iterations(NStepDQNAgent(config))

def ppo_cart_pole():
    config = Config()
    task_fn = lambda log_dir: ClassicalControl('CartPole-v0', max_steps=200, log_dir=log_dir)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, FCBody(state_dim))
    config.discount = 0.99
    config.logger = get_logger()
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 10
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    run_iterations(PPOAgent(config))

def option_critic_cart_pole():
    config = Config()
    game = 'CartPole-v0'
    task_fn = lambda log_dir: ClassicalControl(game, max_steps=200, log_dir=log_dir)
    config.num_workers = 5
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda state_dim, action_dim: OptionCriticNet(
        FCBody(state_dim), action_dim, num_options=2)
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e4))
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.logger = get_logger()
    run_iterations(OptionCriticAgent(config))

def plot():
    import matplotlib.pyplot as plt
    plotter = Plotter()
    names = plotter.load_log_dirs(pattern='.*')
    data = plotter.load_results(names)

    for i, name in enumerate(names):
        x, y = data[i]
        plt.plot(x, y, color=Plotter.COLORS[i], label=name)
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    plt.show()

def action_conditional_video_prediction():
    game = 'PongNoFrameskip-v4'
    prefix = '.'

    # Train an agent to generate the dataset
    # a2c_pixel_atari(game)

    # Generate a dataset with the trained model
    # a2c_model_file = './data/A2CAgent-vanilla-model-%s.bin' % (game)
    # generate_dataset(game, a2c_model_file, prefix)

    # Train the action conditional video prediction model
    acvp_train(game, prefix)


if __name__ == '__main__':
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    set_one_thread()
    select_device(0)

    #dqn_cart_pole()
    a2c_cart_pole()
    # categorical_dqn_cart_pole()
    # quantile_regression_dqn_cart_pole()
    # n_step_dqn_cart_pole()
    # ppo_cart_pole()
    # option_critic_cart_pole()

    # action_conditional_video_prediction()

    #plot()
