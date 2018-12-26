#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''Variations with neuromodulation implemented at Loughborough University.'''

import matplotlib
#matplotlib.use("Pdf")
from deep_rl import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def dqn_pixel_atari(name):
    config = Config()
    config.expType = "dqn_pa" + name
    config.expID = "baseline"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
#    config.max_steps = 2 * 1000000
    config.episode_limit = 100000

    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=config.log_dir)
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
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir=config.log_dir)

    run_episodes(DQNAgent(config))

# L2M
def mod_dqn_pixel_atari_2l(name):
    config = Config()
    config.seed = 123456
    config.expType = "dqn_pa_" + name
    config.expID = "mod2Ldirect"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    #config.max_steps = 5 * 1000000
    config.episode_limit = 100000

    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod2LNatureConvBody_direct_sig())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(log_dir=config.log_dir)
    # config.double_q = True
    config.double_q = False
    run_episodes(L2MAgentYang(config))

def mod_dqn_pixel_atari_3l_2sig(name):
    config = Config()
    config.seed = 1
    config.expType = "dqn_pa_" + name
    config.expID = "mod3Ldirect2sig"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    #config.max_steps = 5 * 1000000
    config.episode_limit = 100000
    config.save_interval = 1000
    config.log_modulation = 1


    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod3LNatureConvBody_direct_2Sig())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(log_dir=config.log_dir)

    # config.double_q = True
    config.double_q = False
    run_episodes(L2MAgentYang(config))

def mod_dqn_pixel_atari_3l_4sig(name):
    config = Config()
    config.seed = 1
    config.expType = "dqn_pa_" + name
    config.expID = "4sigWithLogs"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    #config.max_steps = 5 * 1000000
    config.episode_limit = 100000
    config.save_interval = 20
    config.log_modulation = 1


    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod3LNatureConvBody_direct_4sig())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50
    config.logger = get_logger(log_dir=config.log_dir)

    # config.double_q = True
    config.double_q = False
    run_episodes(L2MAgentYang(config))

def mod_dqn_pixel_atari_3l_relu_shift1(name):
    ''''''
    config = Config()
    config.seed = 1
    config.expType = "dqn_pa_" + name
    config.expID = "RELUplus1"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    #config.max_steps = 5 * 1000000
    config.episode_limit = 100000
    config.save_interval = 50
    config.log_modulation = 1

    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod3LNatureConvBody_direct_relu_shift1())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(log_dir=config.log_dir)

    # config.double_q = True
    config.double_q = False
    run_episodes(L2MAgentYang(config))

def mod_dqn_pixel_atari_3l_relu6_shift05p05(name):
    '''relu 6 but with a min of 0.5, and a shift of 0.5 plus 0.5 to have avg 1 but lower bond 0.5 instead of 0'''
    config = Config()
    config.seed = 1
    config.expType = "dqn_pa_" + name
    config.expID = "RELU-SH1P05"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    #config.max_steps = 5 * 1000000
    config.episode_limit = 100000
    config.save_interval = 100
    config.log_modulation = 1

    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod3LNatureConvBody_direct_relu6_shift05p05())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(log_dir=config.log_dir)

    # config.double_q = True
    config.double_q = False
    run_episodes(L2MAgentYang(config))

def mod_dqn_pixel_atari_3l_diff_relu6_shift05p05(name):
    '''computing differential. relu 6 but with a min of 0.5, and a shift of 0.5 plus 0.5 to have avg 1 but lower bond 0.5 instead of 0'''
    config = Config()
    config.seed = 1
    config.expType = "dqn_pa_" + name
    config.expID = "diff-relu6-s05p05"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    #config.max_steps = 5 * 1000000
    config.episode_limit = 100000
    config.save_interval = 100
    config.log_modulation = 1

    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, diff_relu6_shift05p05())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(log_dir=config.log_dir)

    # config.double_q = True
    config.double_q = False
    run_episodes(L2MAgentYang(config))

def mod_dqn_pixel_atari_3l_relu_shift05p05(name):
    '''relu 6 but with a min of 0.5, and a shift of 0.5 plus 0.5 to have avg 1 but lower bond 0.5 instead of 0'''
    config = Config()
    config.seed = 1
    config.expType = "dqn_pa_" + name
    config.expID = "RELUnon6-SH1P05"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    #config.max_steps = 5 * 1000000
    config.episode_limit = 100000
    config.save_interval = 100
    config.log_modulation = 1

    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod3LNatureConvBody_direct_relu_shift05p05())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(log_dir=config.log_dir)

    # config.double_q = True
    config.double_q = False
    run_episodes(L2MAgentYang(config))

def mod_dqn_pixel_atari_3l_diff(name):
    config = Config()
    config.seed = 1
    config.expType = "dqn_pa_" + name
    config.expID = "mod3L-DIFF"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
#    config.max_steps = 5 * 1000000
    config.episode_limit = 100000
    config.save_interval = 1000
    config.log_modulation = 1


    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: ModDuelingNet(action_dim, Mod3LNatureConvBody_diff_sig())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(log_dir=config.log_dir)

    # config.double_q = True
    config.double_q = False
    run_episodes(L2MAgentYang(config))

def quantile_regression_dqn_pixel_atari(name):
    config = Config()
    config.seed = 1
    config.expType = "qrdqn_pa_" + name
    config.expID = "base"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
#    config.max_steps = 5 * 1000000
    config.episode_limit = 100000
    config.save_interval = 20

    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
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
    config.logger = get_logger(log_dir=config.log_dir)
    config.double_q = False
    config.num_quantiles = 200
    run_episodes(QuantileRegressionDQNAgent(config))

def quantile_regression_dqn_pixel_atari_mod(name):
    '''first version of the QR with modulation, using the relu6s05p05'''
    config = Config()
    config.seed = 1
    config.expType = "qrdqn_pa_" + name
    config.expID = "mod"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
#    config.max_steps = 5 * 1000000
    config.episode_limit = 120000
    config.save_interval = 100
    config.log_modulation = 1

    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                        log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNetMod(action_dim, config.num_quantiles, diff_relu6_shift05p05())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.01, 1e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps= 50000
    config.logger = get_logger(log_dir=config.log_dir)
    config.double_q = False
    config.num_quantiles = 200
    run_episodes(QuantileRegressionDQNAgent_mod(config))

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

def ppo_pixel_atari(name):
    config = Config()
    config.seed = 1
    config.expType = "ppo_pa_" + name
    config.expID = "baseline"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.max_steps = 30 * 1000000
    config.log_modulation = 0

    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, frame_skip=4, history_length=config.history_length, log_dir=config.log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.logger = get_logger(log_dir=config.log_dir)
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

def ppo_pa_mod(name):
    config = Config()
    config.seed = 1
    config.expType = "ppo_pa_" + name
    config.expID = "mrelu6s05p05"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.max_steps = 30 * 1000000
    config.log_modulation = 1

    config.history_length = 4
    task_fn = lambda log_dir: PixelAtari(name, seed=config.seed, frame_skip=4, history_length=config.history_length, log_dir=config.log_dir)
    config.num_workers = 16
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers,
                                              log_dir=config.log_dir)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet_L2M_Mod(
        state_dim, action_dim, Mod3LNatureConvBody_direct_relu6_shift05p05())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.logger = get_logger(log_dir=config.log_dir)
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 4
    config.num_mini_batches = 4
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    run_iterations(PPOAgent_L2M_Mem(config))

if __name__ == '__main__':
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    set_one_thread()
    select_device(1)

    #dqn_pixel_atari('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_2l('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_3l('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_3l_diff('BreakoutNoFrameskip-v4')

#    mod_dqn_pixel_atari_3l_2sig('BreakoutNoFrameskip-v4')
#    mod_dqn_pixel_atari_3l_4sig('BreakoutNoFrameskip-v4')
#    mod_dqn_pixel_atari_3l_relu_shift1('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_3l_relu6_shift05p05('BreakoutNoFrameskip-v4')
    #mod_dqn_pixel_atari_3l_diff_relu6_shift05p05('BreakoutNoFrameskip-v4')

#    mod_dqn_pixel_atari_3l_relu_shift05p05('BreakoutNoFrameskip-v4')

    #mod_dqn_pixel_atari_3l_diff('BreakoutNoFrameskip-v4')

    #quantile_regression_dqn_pixel_atari('BreakoutNoFrameskip-v4')
    quantile_regression_dqn_pixel_atari_mod('BreakoutNoFrameskip-v4')
    #ppo_pixel_atari('BreakoutNoFrameskip-v4')
    ppo_pa_mod('BreakoutNoFrameskip-v4')

#    plot()
