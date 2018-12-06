#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

## cart pole

def dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger()
    # config.double_q = True
    config.double_q = False
    run_episodes(DQNAgent(config))

def nmdqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod1LNatureConvBody_diff())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger()
    # config.double_q = True
    config.double_q = False
    run_episodes(NMDQNAgent(config))

def nmdqn_pixel_atari_v2(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod2LNatureConvBody_diff())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger()
    # config.double_q = True
    config.double_q = False
    run_episodes(NMDQNAgentV2(config))

def nmdqn_pixel_atari_v2(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod2LNatureConvBody_diff())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger()
    # config.double_q = True
    config.double_q = False
    run_episodes(NMDQNAgentV2(config))

def nmdqn_pixel_atari_v3(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod2LNatureConvBody_diffV2())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger()
    # config.double_q = True
    config.double_q = False
    run_episodes(NMDQNAgentV3(config))

def nmdqn_pixel_atari_v4(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod2LNatureConvBody_A_diff())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger()
    # config.double_q = True
    config.double_q = False
    run_episodes(NMDQNAgentV3(config))

def a2c_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.num_workers = 16
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=get_default_log_dir(a2c_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, NatureConvBody())
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = False
    config.gae_tau = 0.97
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.logger = get_logger(file_name=a2c_pixel_atari.__name__, skip=True)
    run_iterations(A2CAgent(config))

def categorical_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(categorical_dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda state_dim, action_dim: \
        CategoricalNet(action_dim, config.categorical_n_atoms, NatureConvBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(0.1))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger()
    config.double_q = False
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    run_episodes(CategoricalDQNAgent(config))

def quantile_regression_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=get_default_log_dir(quantile_regression_dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, NatureConvBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.01, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger()
    config.double_q = False
    config.num_quantiles = 200
    run_episodes(QuantileRegressionDQNAgent(config))

def n_step_dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(n_step_dqn_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.05, 1e6))
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.logger = get_logger()
    run_iterations(NStepDQNAgent(config))

def ppo_pixel_atari(name):
    config = Config()
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(ppo_pixel_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.logger = get_logger(file_name=ppo_pixel_atari.__name__)
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 4
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    run_iterations(PPOAgent(config))

def option_ciritc_pixel_atari(name):
    config = Config()
    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=get_default_log_dir(config.tag))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: OptionCriticNet(NatureConvBody(), action_dim, num_options=4)
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(0.1))
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = 1e8
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    config.logger = get_logger()
    run_iterations(OptionCriticAgent(config))

def dqn_ram_atari(name):
    config = Config()
    config.task_fn = lambda: RamAtari(name, no_op=30, frame_skip=4,
                                      log_dir=get_default_log_dir(dqn_ram_atari.__name__))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, FCBody(state_dim))
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(0.1))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = RescaleNormalizer(1.0 / 128)
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 0
    config.exploration_steps= 100
    config.logger = get_logger()
    config.double_q = True
    # config.double_q = False
    run_episodes(DQNAgent(config))

## continuous control

def ppo_continuous():
    config = Config()
    config.num_workers = 1
    # task_fn = lambda log_dir: Pendulum(log_dir=log_dir)
    # task_fn = lambda log_dir: Bullet('AntBulletEnv-v0', log_dir=log_dir)
    task_fn = lambda log_dir: Roboschool('RoboschoolHopper-v1', log_dir=log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=get_default_log_dir(ppo_continuous.__name__))

    config.network_fn = lambda state_dim, action_dim: GaussianActorCriticNet(
        state_dim, action_dim, actor_body=FCBody(state_dim),
        critic_body=FCBody(state_dim))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    # config.state_normalizer = RunningStatsNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.num_mini_batches = 32
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    config.logger = get_logger()
    run_iterations(PPOAgent(config))

def ddpg_low_dim_state():
    config = Config()
    log_dir = get_default_log_dir(ddpg_low_dim_state.__name__)
    # config.task_fn = lambda **kwargs: Pendulum(log_dir=log_dir)
    # config.task_fn = lambda **kwargs: Bullet('AntBulletEnv-v0', **kwargs)
    config.task_fn = lambda **kwargs: Roboschool('RoboschoolHopper-v1', **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)
    config.max_steps = int(1e6)
    config.evaluation_episodes_interval = int(1e4)
    config.evaluation_episodes = 20

    config.network_fn = lambda state_dim, action_dim: DeterministicActorCriticNet(
        state_dim, action_dim,
        actor_body=FCBody(state_dim, (300, 200), gate=F.tanh),
        critic_body=TwoLayerFCBodyWithAction(state_dim, action_dim, (400, 300), gate=F.tanh),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda action_dim: OrnsteinUhlenbeckProcess(
        size=(action_dim, ), std=LinearSchedule(0.2))
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = get_logger()
    run_episodes(DDPGAgent(config))

def ddpg_pixel():
    config = Config()
    log_dir = get_default_log_dir(ddpg_pixel.__name__)
    config.task_fn = lambda **kwargs: PixelBullet('AntBulletEnv-v0', frame_skip=1,
                                           history_length=4, **kwargs)
    config.evaluation_env = config.task_fn(log_dir=log_dir)

    phi_body=DDPGConvBody()
    config.network_fn = lambda state_dim, action_dim: DeterministicActorCriticNet(
        state_dim, action_dim, phi_body=phi_body,
        actor_body=FCBody(phi_body.feature_dim, (50, ), gate=F.tanh),
        critic_body=OneLayerFCBodyWithAction(phi_body.feature_dim, action_dim, 50, gate=F.tanh),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=16)
    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.max_steps = 1e7
    config.random_process_fn = lambda action_dim: OrnsteinUhlenbeckProcess(
        size=(action_dim, ), std=LinearSchedule(0.2))
    config.min_memory_size = 64
    config.target_network_mix = 1e-3
    config.logger = get_logger(file_name=ddpg_pixel.__name__)
    run_episodes(DDPGAgent(config))

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
    select_device(1)

    #dqn_pixel_atari('BreakoutNoFrameskip-v4')
    #nmdqn_pixel_atari('BreakoutNoFrameskip-v4')
    #nmdqn_pixel_atari_v2('BreakoutNoFrameskip-v4')
    nmdqn_pixel_atari_v4('BreakoutNoFrameskip-v4')
    # a2c_pixel_atari('BreakoutNoFrameskip-v4')
    # categorical_dqn_pixel_atari('BreakoutNoFrameskip-v4')
    # quantile_regression_dqn_pixel_atari('BreakoutNoFrameskip-v4')
    # n_step_dqn_pixel_atari('BreakoutNoFrameskip-v4')
    # ppo_pixel_atari('BreakoutNoFrameskip-v4')
    # option_ciritc_pixel_atari('BreakoutNoFrameskip-v4')
    # dqn_ram_atari('Breakout-ramNoFrameskip-v4')

    # ddpg_low_dim_state()
    # ddpg_pixel()
    # ppo_continuous()

    # action_conditional_video_prediction()

    # plot()
