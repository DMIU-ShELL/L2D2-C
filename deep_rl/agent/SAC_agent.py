#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


#   _________   _____  _________      _____                         __   
#  /   _____/  /  _  \ \_   ___ \    /  _  \    ____   ____   _____/  |_ 
#  \_____  \  /  /_\  \/    \  \/   /  /_\  \  / ___\_/ __ \ /    \   __\
#  /        \/    |    \     \____ /    |    \/ /_/  >  ___/|   |  \  |  
# /_______  /\____|__  /\______  / \____|__  /\___  / \___  >___|  /__|  
#         \/         \/        \/          \//_____/      \/     \/      

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


class SACAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        self.task = config.task_fn()
        print(self.task.state_dim, self.task.action_dim)
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_value_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_value_network.load_state_dict(self.network.state_dict())
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

            # Observe state and selection action from policy network (actor network)
            action = self.network.forward(state).cpu().detach().numpy().flatten()

            # NOTE: Is it possible to use some sort of decaying hyperparameter or epislon-greedy approach to
            # switch from exploration through stochastic actions to deterministic actions to ensure best performance
            # in a lifelong learning scenario? Has this been done already?

            # Add noise from the Gaussian random process function in sac_test.py
            # We use this to introduce a level of stochasticity to the actions which encourages
            # exploration. To be more deterministic in our actions we use the flag.
            if not deterministic:
                action += self.random_process.sample()
                
            # Execute the action 
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
                rewards = tensor(rewards).unsqueeze(-1).to(config.DEVICE)
                next_states = tensor(next_states)
                mask = tensor(1 - terminals).unsqueeze(-1)

                # Compute target actions from forward pass
                # NOTE: torch.no_grad allows us to efficiently perform forward pass/inference when we don't need to compute
                # gradient calculations.
                with torch.no_grad():
                    next_action, next_log_prob = self.network.sample(next_states, reparameterize=True)
                    target_value = self.target_value_network.value(next_states)
                    target_q_value = rewards + config.discount * mask * (target_value - config.alpha * next_log_prob.unsqueeze(-1))


                # Get q values from the two critic networks for the states-actions
                q_value_1, q_value_2 = self.network.q(states, actions)

                # Estimate the state-value function using the state observations
                value = self.network.value(states)


                # Based on SAC implementation found here: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
                # Compute the q-value loss
                qf1_loss = F.mse_loss(q_value_1, target_q_value)
                qf2_loss = F.mse_loss(q_value_2, target_q_value)
                q_value_loss = qf1_loss + qf2_loss

                # Update the two critic networks (q-functions) with gradient descent
                self.network.critic_opt.zero_grad()
                q_value_loss.backward()
                self.network.critic_opt.step()



                # Compute the value loss
                value_loss = F.mse_loss(value, target_q_value.detach())

                # Update the value network
                self.network.value_opt.zero_grad()
                value_loss.backward()
                self.network.value_opt.step()



                # Compute the policy loss
                qf1_pi, qf2_pi = self.network.q(states, next_action)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                policy_loss = (config.alpha * next_log_prob - min_qf_pi).mean()

                # Update the policy (actor) network using gradient ascent
                self.network.actor_opt.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                # Update the target networks
                self.soft_update(self.target_value_network, self.network)
                
            if done:
                break

        return total_reward, steps