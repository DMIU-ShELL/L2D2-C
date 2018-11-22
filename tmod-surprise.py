#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
''' Code to test the surprise architecture'''

import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

# L2M
def mod_dqn_pixel_atari(name):
    config = Config()
    config_mod = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, frame_skip=4, #history_length=config.history_length,
                                        log_dir=get_default_log_dir(dqn_pixel_atari.__name__))
    #config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length)


    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)

    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    # config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())

    #mod network
    #config_mod.network_fn = lambda state_dim, action_dim: NatureConvBody()

    config_mod.network_fn = lambda : CombinedNet(NatureConvBody())

    config_mod.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)

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
    run_episodes(ModDQNAgentSurprise(config,config_mod))

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

if __name__ == '__main__':
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    set_one_thread()
    select_device(1)

    mod_dqn_pixel_atari('BreakoutNoFrameskip-v4')
