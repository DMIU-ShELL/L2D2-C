#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


# ___________________  ________      _____                         __   
# \__    ___/\______ \ \_____  \    /  _  \    ____   ____   _____/  |_ 
#   |    |    |    |  \  _(__  <   /  /_\  \  / ___\_/ __ \ /    \   __\
#   |    |    |    `   \/       \ /    |    \/ /_/  >  ___/|   |  \  |  
#   |____|   /_______  /______  / \____|__  /\___  / \___  >___|  /__|  
#                    \/       \/          \//_____/      \/     \/      

from copy import deepcopy
import multiprocessing as mp
import torch
import torch.nn as nn
import numpy as mp
from queue import Empty
from ..network import network_heads as nethead
from ..shell_modules.mmn.ssmask_utils import set_model_task, consolidate_mask, cache_masks, set_num_tasks_learned, get_mask, set_mask, erase_masks
from .BaseAgent import BaseAgent, BaseContinualLearnerAgent
from ..network.network_bodies import FCBody_SS, DummyBody_CL
from ..utils.torch_utils import random_seed, select_device, tensor
from ..utils.misc import Batcher
from ..component.replay import Replay
import numpy as np
import torch.nn.functional as F


class TD3Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(self.task.action_dim)
        self.total_steps = 0

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        action = self.network.forward(state).cpu().detach().numpy().flatten()
        self.config.state_normalizer.unset_read_only()
        return action
    
    def episode(self, deterministic=False):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.config.state_normalizer(state)

        config = self.config

        steps = 0
        total_reward = 0.0
        while True:
            self.evaluate()
            self.evaluation_episodes()

            #action = self.network.forward(np.stack([state])).cpu().detach().numpy().flatten()
            # you don't need to stack states, since the parallelized env with 1 worker has
            # already added a batch dim for you. So your state space should have the shape:
            # (workers/batch_dim, env_state_dim)
            action = self.network.forward(state).cpu().detach().numpy().flatten()
            if not deterministic:
                action += self.random_process.sample()
                
            next_state, reward, done, info = self.task.step(action)
            next_state = self.config.state_normalizer(next_state)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)

            if not deterministic:
                # remove the worker/batch dim from states, next_states and reward, because
                # the replay is sampled from later on, the sample adds its own batch dim for
                # samples. Since TD3 uses one worker, it's fine to remove the batch dim from
                # states and reward acquired from earlier path of the code.
                state_ = state.squeeze()
                next_state_ = next_state.squeeze()
                reward_ = reward.squeeze()

                self.replay.feed([state_, action, reward_, next_state_, int(done)])
                self.total_steps += 1

            steps += 1
            state = next_state

            # If it's time to update
            if not deterministic and self.replay.size() >= config.min_memory_size:
                # Randomly sample a batch of transitions from the replay buffer
                states, actions, rewards, next_states, terminals = self.replay.sample()
                
                states = tensor(states)
                actions = tensor(actions)
                rewards = tensor(rewards).unsqueeze(-1)
                next_states = tensor(next_states)
                mask = tensor(1 - terminals).unsqueeze(-1)

                # Compute target actions
                a_next = self.target_network(next_states)
                noise = torch.randn_like(a_next).mul(config.td3_noise)
                noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)

                min_a = float(self.task.action_space.low[0])
                max_a = float(self.task.action_space.high[0])

                a_next = (a_next + noise).clamp(min_a, max_a)

                
                # Compute targets
                q_1, q_2 = self.target_network.q(next_states, a_next)
                target = rewards + config.discount * mask * torch.min(q_1, q_2)
                target = target.detach()

                q_1, q_2 = self.network.q(states, actions)
                critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)


                # Update Q-functions by one step of gradient DESCENT (descent used to minimise mean squared error loss between
                # the predicted Q-values q_1 and q_2 on the critic network (Q-functions))
                self.network.zero_grad()
                critic_loss.backward()
                self.network.critic_opt.step()

                # If j (in this case total_steps) mod policy delay = 0
                if self.total_steps % config.td3_delay:
                    # Update policy by one step of gradient ASCENT (ascent used to update the policy (actor network) by maximizing the expected return)
                    # policy loss is the negative mean of the Q-values for the current states and actions
                    action = self.network(states)
                    policy_loss = -self.network.q(states, action)[0].mean()

                    self.network.zero_grad()
                    policy_loss.backward()
                    self.network.actor_opt.step()

                    # Update target networks
                    self.soft_update(self.target_network, self.network)

            if done:
                break

        return total_reward, steps


class TD3ContinualLearnerAgent(BaseContinualLearnerAgent):
    pass

class TD3LLAgent(TD3ContinualLearnerAgent):
    pass

class TD3ShELLDP(TD3LLAgent):
    pass