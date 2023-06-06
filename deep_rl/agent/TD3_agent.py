#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


#___________.__             _________                __  .__                                        _____                         __   
#\__    ___/|  |__   ____   \_   ___ \  ____   _____/  |_|__| ____   ____  __ __   ____   ______   /  _  \    ____   ____   _____/  |_ 
#  |    |   |  |  \_/ __ \  /    \  \/ /  _ \ /    \   __\  |/    \ /  _ \|  |  \_/ __ \ /  ___/  /  /_\  \  / ___\_/ __ \ /    \   __\
#  |    |   |   Y  \  ___/  \     \___(  <_> )   |  \  | |  |   |  (  <_> )  |  /\  ___/ \___ \  /    |    \/ /_/  >  ___/|   |  \  |  
#  |____|   |___|  /\___  >  \______  /\____/|___|  /__| |__|___|  /\____/|____/  \___  >____  > \____|__  /\___  / \___  >___|  /__|  
#                \/     \/          \/            \/             \/                   \/     \/          \//_____/      \/     \/      
#

from copy import deepcopy
import multiprocessing as mp
import torch
#import torch.nn as nn
#import numpy as mp
#from queue import Empty
#from ..network import network_heads as nethead
#from ..shell_modules.mmn.ssmask_utils import set_model_task, consolidate_mask, cache_masks, set_num_tasks_learned, get_mask, set_mask, erase_masks
from .BaseAgent import BaseAgent, BaseContinualLearnerAgent
#from ..network.network_bodies import FCBody_SS, DummyBody_CL
from ..utils.torch_utils import random_seed, select_device, tensor
#from ..utils.misc import Batcher
#from ..component.replay import Replay
import numpy as np
import torch.nn.functional as F

#from ..network import *
#from ..component import *
#from .BaseAgent import *
#import torchvision


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
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        action = self.network.predict(state, to_numpy=True).flatten()
        self.config.state_normalizer.unset_read_only()
        return action

    #def eval_step(self, state):
    #    self.config.state_normalizer.set_read_only()
    #    state = self.config.state_normalizer(state)
    #    action = self.network(state)
    #    self.config.state_normalizer.unset_read_only()
    #    return action.cpu().detach().numpy()

    def episode(self, deterministic=False):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        steps = 0
        total_reward = 0.0
        while True:
            self.evaluate()
            self.evaluation_episodes()

            action = self.network.forward(np.stack([self.state]))
            action = action.cpu().detach().numpy()
            action = action.flatten()
            print(action)
            if not deterministic:
                print(self.random_process.sample())
                action += self.random_process.sample()
            next_state, reward, done, info = self.task.step(action)
            next_state = self.config.state_normalizer(next_state)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)

            if not deterministic:
                self.replay.feed([self.state, action, reward, next_state, int(done)])
                self.total_steps += 1

            steps += 1
            self.state = next_state

            if not deterministic and self.replay.size() >= config.min_memory_size:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = tensor(states)
                actions = tensor(actions)
                rewards = tensor(rewards).unsqueeze(-1)
                next_states = tensor(next_states)
                mask = tensor(1 - terminals).unsqueeze(-1)


                ##################################################################
                #                         TD-DDPG ALGORITHM
                ##################################################################

                a_next = self.target_network(next_states)
                noise = torch.randn_like(a_next).mul(config.td3_noise)
                noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)

                min_a = float(self.task.action_space.low[0])
                max_a = float(self.task.action_space.high[0])
                a_next = (a_next + noise).clamp(min_a, max_a)

                q_1, q_2 = self.target_network.q(next_states, a_next)
                target = rewards + config.discount * mask * torch.min(q_1, q_2)
                target = target.detach()

                q_1, q_2 = self.network.q(states, actions)
                critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

                self.network.zero_grad()
                critic_loss.backward()
                self.network.critic_opt.step()

                if self.total_steps % config.td3_delay:
                    action = self.network(states)
                    policy_loss = -self.network.q(states, action)[0].mean()

                    self.network.zero_grad()
                    policy_loss.backward()
                    self.network.actor_opt.step()

                    self.soft_update(self.target_network, self.network)

                if done:
                    break
            
            return total_reward, steps
            
        
        
        '''if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            action = self.network(self.state)
            action = action.cpu().detach().numpy()
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        #self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        experiences = list(zip(self.state, action, reward, next_state, done))
        self.replay.feed_batch(experiences)
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)


            ##################################################################
            #                         TD-DDPG ALGORITHM
            ##################################################################

            a_next = self.target_network(next_states)
            noise = torch.randn_like(a_next).mul(config.td3_noise)
            noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)

            min_a = float(self.task.action_space.low[0])
            max_a = float(self.task.action_space.high[0])
            a_next = (a_next + noise).clamp(min_a, max_a)

            q_1, q_2 = self.target_network.q(next_states, a_next)
            target = rewards + config.discount * mask * torch.min(q_1, q_2)
            target = target.detach()

            q_1, q_2 = self.network.q(states, actions)
            critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            if self.total_steps % config.td3_delay:
                action = self.network(states)
                policy_loss = -self.network.q(states, action)[0].mean()

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network)
            ##################################################################'''


def TD3ContinualLearnerAgent(BaseContinualLearnerAgent):
    return