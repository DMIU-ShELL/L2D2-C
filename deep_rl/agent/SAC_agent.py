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
import gym
from torch.distributions import Normal
import traceback


class SACAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        self.task = config.task_fn()
        print(self.task.state_dim, self.task.action_dim)
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.network.to(config.DEVICE)
        self.target_value_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_value_network.load_state_dict(self.network.state_dict())
        self.target_value_network.to(config.DEVICE)
        self.data_buffer = config.replay_fn()
        self.data_buffer = Replay(memory_size=int(1e6), batch_size=64)
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

                self.data_buffer.feed([state_, action, reward_, int(done), next_state_])
                self.total_steps += 1

            steps += 1
            state = next_state

            # If it's time to update (Until then we are filling the replay buffer)
            if not deterministic and self.data_buffer.size() >= config.min_memory_size:
                # Randomly sample a batch of transitions from the replay buffer
                states, actions, rewards, terminals, next_states = self.data_buffer.sample()
                

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
    

# SAC with lifelong learning setup
"""class SACContinualLearnerAgent_orig(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)
        self.config = config

        self.task = None if config.task_fn is None else config.task_fn()
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task
        tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
        tasks = [tasks_[task_id] for task_id in config.task_ids]
        self.config.cl_tasks_info = tasks
        label_dim = None if not self.config.use_task_label else len(tasks[0]['task_label'])

        self.task_label_dim = label_dim

        random_seed(9157)
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.network.to(config.DEVICE)

        self.target_value_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_value_network.load_state_dict(self.network.state_dict())
        self.target_value_network.to(config.DEVICE)

        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(self.task.action_dim)

        random_seed(config.seed)

        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

        self.running_episodes_rewards = [[] for _ in range(config.num_workers)]
        self.iteration_rewards = np.zeros(config.num_workers)

        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.layers_output = None

        self.curr_train_task_label = None
        self.curr_eval_task_label = None

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
    
    def step(self, deterministic=False):
        config = self.config

        if self.states is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = self.config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.tasks[0].action_space.sample()]     # Quick workaround for ParallelizedTask() with single_process=True. Original: action = [self.task.action_space.sample()]

        else:
            action = self.network(self.state)
            action = action.cpu().detach().numpy()
            action += self.random_process.sample()

        action = np.clip(action, self.task.tasks[0].action_space.low, self.task.tasks[0].action_space.high)     # Quick workaround for ParallelizedTask() with single_process=True. Original: action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        experiences = list(zip(self.state, action, reward, next_state, done))
        self.replay.feed_batch(experiences)
        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1


        # Setup logging
        grad_norm_log = []
        policy_loss_log = []
        value_loss_log = []
        log_probs_log = []
        entropy_log = []
        target_q_value_log = []


        if self.replay.size() >= config.warm_up:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)
            
            a_next = self.target_network(next_states)
            noise = torch.randn_like(a_next).mul(config.td3_noise)
            noise = noise.clamp(-config.td3_noise_clip, config.td3_noise.clip)

            min_a = float(self.task.tasks[0].action_space.low[0])       # Quick workaround for ParallelizedTask() with single_process=True. Original: min_a = float(self.task.tasks[0].action_space.low[0])
            max_a = float(self.task.tasks[0].action_space.high[0])      # Quick workaround for ParallelizedTask() with single_process=True. Original: max_a = float(self.task.tasks[0].action_space.high[0])
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

            # If it's time to update (Until then we are filling the replay buffer)
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

                # Logging
                log_probs_log.append(next_log_prob.detach().cpu().numpy().mean())
                #entropy_log.append(entropy_loss.detach().cpu().numpy().mean())
                target_q_value_log.append(target_q_value.detach().cpu().numpy().mean())
                policy_loss_log.append(policy_loss.detach().cpu().numpy())
                value_loss_log.append(value_loss.detach().cpu().numpy())

                # Update the target networks
                self.soft_update(self.target_value_network, self.network)
                
            if done:
                break

        #return total_reward, steps
        return {'grad_norm': grad_norm_log, 'policy_loss': policy_loss_log, \
            'value_loss': value_loss_log, 'log_prob': log_probs_log, \
            'sac_target_q_value': target_q_value_log}

class SACContinualLearnerAgent_org2(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)
        self.config = config

        self.task = None if config.task_fn is None else config.task_fn()
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task

        tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
        tasks = [tasks_[task_id] for task_id in config.task_ids]
        self.config.cl_tasks_info = tasks

        label_dim = None if not self.config.use_task_label else len(tasks[0]['task_label'])
        self.task_label_dim = label_dim

        self.logger = self.config.logger


        random_seed(config.backbone_seed)
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.network.to(config.DEVICE)

        self.q_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.q_network.to(config.DEVICE)
        
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.to(config.DEVICE)

        self.target_q_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.to(config.DEVICE)


        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(self.task.action_dim)

        random_seed(config.seed)


        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src, tau):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = np.stack([self.config.state_normalizer(state)])
        action = self.network.forward(state).cpu().detach().numpy().flatten()
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config

        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.tasks[0].action_space.sample()]
        else:
            action, _ = self.network.sample(self.state)
            action = action.cpu().detach().numpy()
            action += self.random_process.sample()

        action = np.clip(action, self.task.tasks[0].action_space.low, self.task.tasks[0].action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        
        
        #self.record_online_return(info)
        # Record online return
        '''if isinstance(info[0], dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps, ret))
        
        elif isinstance(info[0], tuple):
            for offset, info_ in enumerate(info):
                ret = info_['episodic_return']
                if ret is not None:
                    self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                    self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        else:
            raise NotImplementedError'''
        

        
        reward = self.config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            transitions = self.replay.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            with torch.no_grad():
                next_action, next_log_pi = self.network.sample(next_states)
                next_q1 = self.q_network(next_states, next_action)
                next_q2 = self.target_q_network(next_states, next_action)
                min_next_q = torch.min(next_q1, next_q2) - config.alpha * next_log_pi
                expected_q = rewards + config.discount * mask * min_next_q

            q1 = self.q_network(states, actions)
            q2 = self.target_q_network(states, actions)

            q1_loss = (q1 - expected_q).pow(2).mul(0.5).sum(-1).mean()
            q2_loss = (q2 - expected_q).pow(2).mul(0.5).sum(-1).mean()
            critic_loss = q1_loss + q2_loss

            self.q_network.zero_grad()
            critic_loss.backward()
            self.q_network_opt.step()

            with torch.no_grad():
                new_action, new_log_pi = self.network.sample(states)
                new_q1 = self.q_network(states, new_action)
                new_q2 = self.target_q_network(states, new_action)
                min_new_q = torch.min(new_q1, new_q2)
                expected_new_q = min_new_q - config.alpha * new_log_pi
                policy_loss = (config.alpha * new_log_pi - expected_new_q).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network_opt.step()

            self.soft_update(self.target_network, self.network, config.target_network_mix)
            self.soft_update(self.target_q_network, self.q_network, config.target_network_mix)

class SACContinualLearnerAgent_orig3(BaseContinualLearnerAgent):
    def __init__(self, config):
        super(SACContinualLearnerAgent, self).__init__(config)
        self.config = config
        self.task = None if config.task_fn is None else config.task_fn()
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task
        tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
        tasks = [tasks_[task_id] for task_id in config.task_ids]
        del tasks_
        self.config.cl_tasks_info = tasks

        self.task_label_dim = None if not config.use_task_label else len(tasks[0]['task_label'])

        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, self.task_label_dim)
        _params = list(self.network.parameters())
        self.opt = config.optimizer_fn(_params, config.lr)
        self.total_steps = 0

        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

        self.data_buffer = Replay(memory_size=int(1.5 * 1e2), batch_size=130)

        self.curr_train_task_label = None
        self.curr_eval_task_label = None

    def step(self):
        config = self.config
        rollout = []
        states = self.states

        for _ in range(config.rollout_length):
            actions, log_probs = self.select_action(states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())

            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)

            # Save data to buffer for the off-policy update
            self.data_buffer.feed_batch([states, actions.cpu(), rewards, terminals, next_states])

            rollout.append([states, actions, log_probs, rewards, 1 - terminals])
            states = next_states

        self.states = states
        self.off_policy_update(self.data_buffer)

        return {'average_reward': np.mean(self.episode_rewards)}

    def select_action(self, states):
        with torch.no_grad():
            policy_outputs, _ = self.network(states)
            action_mean, action_log_std = policy_outputs
            normal = Normal(action_mean, action_log_std.exp())
            actions = normal.rsample()
            log_probs = normal.log_prob(actions)
        return actions, log_probs

    def off_policy_update(self, replay_buffer):
        batch_size = 64
        if len(replay_buffer) < batch_size:
            return

        # Sample a mini-batch from the replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(1 - done)

        # Compute the target Q values
        with torch.no_grad():
            next_action = self.actor(next_state)
            target_Q1 = self.target_critic1(next_state, next_action)
            target_Q2 = self.target_critic2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + self.discount * done * target_Q

        # Critic loss
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)
        critic1_loss = nn.functional.mse_loss(current_Q1, target_Q)
        critic2_loss = nn.functional.mse_loss(current_Q2, target_Q)

        # Update the critic networks
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy update
        if len(replay_buffer) % 2 == 0:
            # Actor loss
            actor_action = self.actor(state)
            Q1 = self.critic1(state, actor_action)
            Q2 = self.critic2(state, actor_action)
            actor_loss = -torch.mean(torch.min(Q1, Q2))

            # Update the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target critics
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)"""

class SACContinualLearnerAgent(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)
        self.config = config

        self.task = None if config.task_fn is None else config.task_fn()
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task
        tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
        tasks = [tasks_[task_id] for task_id in config.task_ids]
        del tasks_
        self.config.cl_tasks_info = tasks
        label_dim = None if not config.use_task_label else len(tasks[0]['task_label'])
        #label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label']) #CHANGE THAT FOR THE
        self.task_label_dim = label_dim


        random_seed(config.backbone_seed)
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.network.to(config.DEVICE)
        self.target_value_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_value_network.load_state_dict(self.network.state_dict())
        self.target_value_network.to(config.DEVICE)
        
        #self.data_buffer = config.replay_fn()
        self.data_buffer = Replay(memory_size=int(1e6), batch_size=64)

        self.random_process = config.random_process_fn(self.task.action_dim)
        self.total_steps = 0
        random_seed(config.seed)


        self.rewards = [[] for _ in range(config.num_workers)]
        self.episode_returns = np.zeros(config.num_workers)
        #self.iteration_rewards = np.zeros(config.num_workers) # Compatibility with the rest of the repository

        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)


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
        state = self.states

        config = self.config

        steps = 0
        #total_reward = 0.0
        policy_loss_log = []
        value_loss_log = []
        q_value_loss_log = []
        log_probs_log = []
        #noise = 0
        self.episode_returns = np.zeros(config.num_workers)
        #self.iteration_rewards = np.zeros(config.num_workers)
        try:
            while True:
                #self.evaluate()
                #self.evaluation_episodes()

                # Observe state and selection action from policy network (actor network)
                action = self.network.forward(state).cpu().detach().numpy().flatten()

                # NOTE: Is it possible to use some sort of decaying hyperparameter or epislon-greedy approach to
                # switch from exploration through stochastic actions to deterministic actions to ensure best performance
                # in a lifelong learning scenario? Has this been done already?

                # Add noise from the Gaussian random process function in sac_test.py
                # We use this to introduce a level of stochasticity to the actions which encourages
                # exploration. To be more deterministic in our actions we use the flag.
                if not deterministic:
                    #noise = self.random_process.sample()
                    #action += noise  # Add stochasticity for SAC
                    action += self.random_process.sample()
                    
                # Execute the action 
                next_state, reward, terminals, info = self.task.step(action)
                next_state = self.config.state_normalizer(next_state)
                for i, r in enumerate(reward):
                    self.episode_returns[i] += r
                    #self.iteration_rewards = self.episode_returns
                
                #print(self.episode_returns, reward)
                reward = self.config.reward_normalizer(reward)

                if not deterministic:
                    # remove the worker/batch dim from states, next_states and reward, because
                    # the replay is sampled from later on, the sample adds its own batch dim for
                    # samples. Since TD3 and SAC uses one worker, it's fine to remove the batch dim from
                    # states and reward acquired from earlier path of the code.
                    state_ = state.squeeze()
                    next_state_ = next_state.squeeze()
                    reward_ = reward.squeeze()

                    self.data_buffer.feed([state_, action, reward_, int(terminals), next_state_])
                    self.total_steps += 1

                steps += 1
                state = next_state

                # If it's time to update (Until then we are filling the replay buffer)
                if not deterministic and self.data_buffer.size() >= config.min_memory_size:
                    # Randomly sample a batch of transitions from the replay buffer
                    states, actions, rewards, terminals, next_states = self.data_buffer.sample()
                    

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
                        #print('next_log_prob:', next_log_prob)
                        log_probs_log.append(next_log_prob.detach().cpu().numpy().mean().item())


                    # Get q values from the two critic networks for the states-actions
                    q_value_1, q_value_2 = self.network.q(states, actions)

                    # Estimate the state-value function using the state observations
                    value = self.network.value(states)


                    # Based on SAC implementation found here: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
                    # Compute the q-value loss
                    qf1_loss = F.mse_loss(q_value_1, target_q_value)
                    qf2_loss = F.mse_loss(q_value_2, target_q_value)
                    q_value_loss = qf1_loss + qf2_loss
                    #print('q_value_loss:', q_value_loss)
                    q_value_loss_log.append(q_value_loss.detach().cpu().numpy())

                    # Update the two critic networks (q-functions) with gradient descent
                    self.network.critic_opt.zero_grad()
                    q_value_loss.backward()
                    self.network.critic_opt.step()



                    # Compute the value loss
                    value_loss = F.mse_loss(value, target_q_value.detach())
                    #print('value_loss:', value_loss_log)
                    value_loss_log.append(value_loss.detach().cpu().numpy())

                    # Update the value network
                    self.network.value_opt.zero_grad()
                    value_loss.backward()
                    self.network.value_opt.step()



                    # Compute the policy loss
                    qf1_pi, qf2_pi = self.network.q(states, next_action)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    policy_loss = (config.alpha * next_log_prob - min_qf_pi).mean()
                    #print('policy_loss:', policy_loss)
                    policy_loss_log.append(policy_loss.detach().cpu().numpy())

                    # Update the policy (actor) network using gradient ascent
                    self.network.actor_opt.zero_grad()
                    policy_loss.backward()
                    self.network.actor_opt.step()


                    # Update the target networks
                    self.soft_update(self.target_value_network, self.network)
                    
                print(terminals)
                print(len(terminals))
                print(bool(terminals))
                print(terminals[0])
                if bool(terminals):
                    break
        except Exception as e:
            traceback.print_exc()

        # Return the training statistics as a dictionary
        training_stats = {
            "policy_loss": policy_loss_log,
            "value_loss": value_loss_log,
            "q_value_loss": q_value_loss_log,
            "log_probs": log_probs_log
            #"noise_value": noise
        }

        return training_stats
    

class SACBaseAgent(SACContinualLearnerAgent):
    '''SAC continual learning agent baseline (experiences catastrophic forgetting). Uses simplified task_*_start and task_*_end methods to override masking.'''
    def __init__(self, config):
        SACContinualLearnerAgent.__init__(self, config)

    def task_train_start(self, task_label):
        self.curr_train_task_label = task_label
        return

    def task_train_end(self):
        self.curr_train_task_label = None
        return

    def task_eval_start(self, task_label):
        self.curr_eval_task_label = task_label
        return

    def task_eval_end(self):
        self.curr_eval_task_label = None
        return

class SACLLAgent(SACContinualLearnerAgent):
    def __init__(self, config):
        SACContinualLearnerAgent.__init__(self, config)
        self.seen_tasks = {} # contains task labels that agent has experienced so far
        self.new_task = False
        self.curr_train_task_label = None
        self.curr_eval_task_label = None

    def get_seen_tasks(self):
        '''A getter method for deriving which tasks the DMIU agent has encountered.'''
        return self.seen_tasks
     
    def _label_to_idx(self, task_label):
        eps = 1e-5
        found_task_idx = None
        for task_idx, seen_task_label in self.seen_tasks.items():
            if np.linalg.norm((task_label - seen_task_label), ord=2) < eps:
                found_task_idx = task_idx
                break
        return found_task_idx
    
    def erase_memory(self, current_task_label):
        self.seen_tasks = {0: current_task_label}
        erase_masks(self.network, self.config.DEVICE)

    def task_train_start(self, task_label):
        task_idx = self._label_to_idx(task_label)
        if task_idx is None:
            task_idx = len(self.seen_tasks)
            self.seen_tasks[task_idx] = task_label
            self.new_task = True
            set_model_task(self.network, task_idx, new_task=True)
        else:
            set_model_task(self.network, task_idx)
        self.curr_train_task_label = task_label
        return
    
    def task_train_end(self):
        # NOTE, comment/uncomment alongside a block of code in '_forward_mask_linear_comb' method in
        # MultitaskMaskLinear and MultiMaskLinearSparse classes
        consolidate_mask(self.network)

        self.curr_train_task_label = None
        cache_masks(self.network)
        if self.new_task:
            set_num_tasks_learned(self.network, len(self.seen_tasks))
        self.new_task = False # reset flag
        return
    
    def task_eval_start(self, task_label):
        self.network.eval()
        task_idx = self._label_to_idx(task_label)
        if task_idx is None:
            # agent has not been trained on current task
            # being evaluated. therefore use a random mask
            # TODO: random task hardcoded to the first learnt
            # task/mask. update this later to use a random
            # previous task, or implementing a way for
            # agent to use an ensemble of different mask
            # interally for the task not yet seen.
            task_idx = 0
        set_model_task(self.network, task_idx)
        self.curr_eval_task_label = task_label
        return
    
    def task_eval_end(self):
        self.curr_eval_task_label = None
        self.network.train()
        # resume training the model on train task label if training
        # was on before running evaluations.
        if self.curr_train_task_label is not None:
            task_idx = self.label_to_idx(self.curr_train_task_label)
            set_model_task(self.network, task_idx)
        return
    
class SACShellAgent(SACLLAgent):
    def __init__(self, config):
        SACLLAgent.__init__(self, config)
        _mask = get_mask(self.network, task=0)
        self.mask_info = {}
        for key, value in _mask.items():
            self.mask_info[key] = tuple(value.shape)
        model_mask_dim = 0
        for k, v in self.mask_info.items():
            model_mask_dim += np.prod(v)
        self.model_mask_dim = model_mask_dim

    def _select_mask(self, masks, ensemble=False):
        found_mask = None
        if ensemble:
            raise NotImplementedError
        else:
            for mask in masks:
                if mask is not None:
                    found_mask = mask
                    break
        return found_mask
    
    def label_to_mask(self, task_label):
        task_idx = self._label_to_idx(task_label)
        # get task mask
        if task_idx is None:
            mask = None
        else:
            # function from deep_rl/shell_modules/mmn/ssmask_utils.py
            mask = get_mask(self.network, task_idx)
            mask = self.mask_to_vec(mask)
        return mask
    
    def distil_task_knowledge(self, masks):
        # TODO: fix algorithm. Current solution involves selecting the first mask
        # that is not None and using it as the knowledge from which the agent can
        # continue training on current task.
        task_label = self.curr_train_task_label
        #task_label = self.task.get_task()['task_label'] # use this to pass label from outside fn
        task_idx = self._label_to_idx(task_label)

        masks = [self.vec_to_mask(mask.to(self.config.DEVICE)) for mask in masks]
        mask = self._select_mask(masks)

        if mask is not None:
            # function from deep_rl/shell_modules/mmn/ssmask_utils.py
            set_mask(self.network, mask, task_idx)
            return True
        else:
            return False
        
    def distil_task_knowledge_single(self, mask, task_label):
        # New distil task knowledge algorithm
        # this function receives only one mask
        # which is the best mask.
        #print(mask)

        #task_albel = self.curr_train_task_label
        task_idx = self._label_to_idx(task_label)

        # Process the single maks as opposed to multiple
        mask = self.vec_to_mask(mask.to(self.config.DEVICE))

        if mask is not None:
            set_mask(self.network, mask, task_idx)
            return True
        else:
            return False
        
    def distil_task_knowledge_single_eval(self, mask, task_label):
        # New distil task knowledge algorithm
        # this function receives only one mask
        # which is the best mask.
        #print(mask)

        #task_label = self.curr_eval_task_label
        #print(task_label)
        task_idx = self._label_to_idx(task_label)
        #print(task_idx)

        # Process the single mask as opposed to multiple
        mask = self.vec_to_mask(mask.to(self.config.DEVICE))

        if mask is not None:
            set_mask(self.network, mask, task_idx)
            return True
        else:
            return False


    def mask_to_vec(self, dict_mask):
        with torch.no_grad():
            vec_mask = torch.zeros(self.model_mask_dim,)
            start = 0
            stop = None
            for k, v in dict_mask.items():
                stop = start + np.prod(v.shape)
                vec_mask[start : stop] = v.reshape(-1,)
                start = stop
        return vec_mask

    def vec_to_mask(self, vec_mask):
        dict_mask = {}
        start = 0
        stop = None
        for key, value in self.mask_info.items():
            stop = start + np.prod(value)
            dict_mask[key] = vec_mask[start : stop].reshape(*value)
            start = stop
        return dict_mask

class SACDetectShell(SACShellAgent):
    '''Detect L2D2-C agent implementation. Uses modified methods to support use of the detect module.'''
    def __init__(self, config):
        SACShellAgent.__init__(self, config)

        # Saptarshi: Updating the seen_tasks dictionary to be use the SyncManager() internal server solution.
        # this will allow us to use the seen_tasks dictionary across our entire parallelised system.
        self.seen_tasks = config.seen_tasks
        self.current_task_key = 0



        ###############################################################################
        # Detect Module Attributes

        '''
        # Create a reference for the Wasserstein Embeddings
        torch.manual_seed(98)
        reference = torch.rand(500, self.task.state_dim)
        '''

        # Variable for storing the action space size of the task for using it to 
        # convert the actions to one-hot vectors.
        if isinstance(self.task.tasks[0].action_space, gym.spaces.Discrete):
            self.task_action_space_size = self.task_action_space_size = self.task.tasks[0].action_space.n
        else:
            self.task_action_space_size = self.task_action_space_size = self.task.tasks[0].action_space.shape[0]

        #self.task_action_space_size = self.task.tasks[0].action_space.n

        # Variable for storing the detect reference number.
        self.detect_reference_num = config.detect_reference_num

        # Variable for storing the number of samples hyperperameter of the detect module.
        self.detect_num_samples = config.detect_num_samples

        # Varible for checking for dishiding wether the agent has encountered a new task or not.
        self.emb_dist_threshold = config.emb_dist_threshold

        # Variable for storing the frequency of the detect module activation.
        self.detect_module_activation_frequency = config.detect_module_activation_frequency

        # Create a list for saving the calculated embeddings
        self.encountered_task_embs = []

        # Varible for storing the current embedding label as an attribute of the agent itself.
        self.current_task_emb =  None #self.task.get_task()['task_emb']

        #Varibles for tracking the current task embedding, influenced by the two varibles above, and working at a same fashion.
        self.curr_train_task_emb = None
        self.curr_eval_task_emb = None

        self.new_task_emb = None


        ###############################################################################
        # Precalculate the embedding size based on the reference and the network observation size.
        tmp_state_obs  = self.task.reset()
        tmp_state_obs = config.state_normalizer(tmp_state_obs)
        observation_size = tmp_state_obs.shape[1]

        # Initialise detect module within the agent
        self.detect = config.detect_fn(self.detect_reference_num, observation_size, self.detect_num_samples)

        # Variable for saving the size of the task embedding that the detect module has produced.
        # Initially we store the precalculated embedding size
        self.detect.set_reference(observation_size, self.detect_reference_num, self.task.action_dim)
        self.task_emb_size  = self.detect.precalculate_embedding_size(self.detect_reference_num, observation_size, self.task.action_dim)
        self.current_task_emb = torch.zeros(self.get_task_emb_size())
        self.new_task_emb =  torch.zeros(self.get_task_emb_size())      # FIXME: We don't use this! Consider removal.

        # Debugging outputs.
        print(f'Observation size: {observation_size}, Task embedding size: {self.task_emb_size}')
        print(f'Task action space size: {self.task_action_space_size}')


    ###############################################################################
    # Modified agent methods to accomodate changes for detect module integration
    def _embedding_to_idx(self, task_embedding):
        '''Private method that checks wether the agent has already seen the task with the
        embedding given before or not. It checks this by searching the dictionary of seen
        tasks and compering the given embedding with the stored emebddings of each stored
        task. If an an embedding with high similarity (based on the distance threshold) is
        found then it return the index of the task for this embedding.
        
        Important NOTE: The index for each of the seen tasks that the agent has encountered
        is different from the original index that the tasks already have in their info 
        dictonary which is being set upon their creation (from the task class). When a task
        is registered in the seen_tasks dictonary of the agent the agent automatically creates
        an unique index for this specific task, which is based on which turn the agent first
        encounters this specific task.'''

        found_task_idx = None
        for task_idx, task_dict in self.seen_tasks.items():
            seen_task_embedding = task_dict['task_emb']
            print(f'Seen task embedding: {seen_task_embedding}, embedding: {task_embedding}')
            cosine_similarity = F.cosine_similarity(task_embedding, seen_task_embedding, dim=0)
            self.config.logger.info(cosine_similarity)
            if cosine_similarity > 0.4:#np.linalg.norm(task_embedding - seen_task_embedding) < self.emb_dist_threshold:
                found_task_idx = task_idx
                break
        return found_task_idx
        
    def task_train_start_emb(self, task_embedding):
        '''Method for starting the training procedure upon a new Detected task form the Detect Module.
        It is based on the "task_train_start()" method.'''

        # Get the task idx (key) for if the task record exists (based on embedding similarity)
        self.current_task_key = self._embedding_to_idx(task_embedding)

        if self.current_task_key is None:
            # If no similar embedding record was found then its a new task
            self.current_task_key = len(self.seen_tasks)                                        # Generate an internal task index for new task
            
            # Create the dictionary key-value pair for the new task
            self.update_seen_tasks(
                embedding=task_embedding, 
                reward=0,
                label=self.task.get_task()['task_label']
            )
            #self.seen_tasks[self.current_task_key] = {
            #    'task_emb': task_embedding,
            #    'reward': 0,
            #    'ground_truth': self.task.get_task()['task_label']
            #}

            self.new_task = True                                                                # Set the new_task flag to True
            set_model_task(self.network, self.current_task_key, new_task=True)                  # Set the new task mask inside the model

        else:
            # Set model to use the existing task mask.
            set_model_task(self.network, self.current_task_key)

        return
    
    def task_train_end_emb(self):
        '''Method for stopping the training upon a specific task. It is being used when
        the Detect Module perceives a task change, in order to stop training for the old
        task and start training for the new one. Inspired from the original "task_train_end()"
        to work with embeddings instead of labels.'''
        # NOTE, comment/uncomment alongside a block of code in `_forward_mask_lnear_comb` method in
        # MultitaskMaskLinear and MultiMaskLinearSparse classes
        consolidate_mask(self.network)

        self.current_task_emb = None
        cache_masks(self.network)
        
        if self.new_task:
            set_num_tasks_learned(self.network, len(self.seen_tasks))

        self.new_task = False # reset flag
        return

    def embedding_to_mask(self, task_embedding):
        '''A method for finding the correspoding mask for the task tha it is currently encountered
        based on the agents task/mask registry (seen tasks dictionary)'''
        task_idx = self._embedding_to_idx(task_embedding)
        #get the stored correspoding mask for that specific task
        if task_idx is None:
            mask = None
        else:
            #function from ssmask_utils.py
            mask = get_mask(self.network, task_idx)
            mask = self.mask_to_vec(mask)
        return mask
    
    def idx_to_mask(self, task_idx):
        '''A method for finding the correspoding mask for the task tha it is currently encountered
        based on the agents task/mask registry (seen tasks dictionary)'''
        #get the stored correspoding mask for that specific task
        if task_idx is None:
            mask = None
        else:
            #function from ssmask_utils.py
            mask = get_mask(self.network, task_idx)
            mask = self.mask_to_vec(mask)
        return mask



    ###############################################################################
    # Mask linear combination methods.
    def linear_combination(self, masks):
        """
        linearly combines incoming masks from the network
        """
        # Get current training mask
        _subnet = self.idx_to_mask(self.current_task_key)

        _subnets = [masks[idx].detach() for idx in range(len(masks))]
        #assert len(_subnets) > 0, 'an error occured'
        #_betas = self.shared_betas[len(masks), 0:len(masks)]    # 2D beta parameter vector
        #_betas = self.shared_betas[0:len(masks)]   # 1D beta parameter vector

        _betas = torch.zeros(len(_subnets)) # beta parameters with equal probability. We can manually set the weights on the beta parameters.
        _betas = torch.softmax(_betas, dim=-1)
        _subnets.append(_subnet)
        #assert len(_betas) == len(_subnets), 'an error ocurred'
        #_subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]  # beta coefficients applied
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        #_subnet_linear_comb = torch.stack(_subnets, dim=0).mean(dim=0)  # equivalent to setting beta parameter as 1/num(masks)
        return self._subnet_class.apply(_subnet_linear_comb)
    
    def consolidate_incoming(self, masks):
        with torch.no_grad():
            new_mask = self.linear_combination(masks)
            self.distil_task_knowledge_embedding(new_mask)

            #set_mask(new_mask, self.current_task_key)



    ###############################################################################
    # Methods for detect module
    def extract_sar(self):
        '''Function for extracting the SAR data from the Replay
        Buffer in order to feed the Detect Module'''

        buffer_data = self.data_buffer.sample()
        #print("RB_DATA:", buffer_data[0])
        #print(type(buffer_data[0]))
        #print("RB_DATA_shape:", len(buffer_data))
        sar_data = []
        #ql = []
        for tpl in buffer_data:
            tmp0 = tpl[:3]
            tmp1 = np.array(tmp0[1])
            tmp1 = tmp1.reshape(1,)
            tmp2 = np.array(tmp0[2])
            tmp2 = tmp2.reshape(1,)
            tmp3 = np.concatenate([tmp0[0].ravel(), tmp1, tmp2])
            sar_data.append(tmp3)

        '''for tpl in sar_data:
            tmp1 = np.array(tpl[1])
            tmp1 = tmp1.reshape(1,)
            tmp2 = np.array(tpl[2])
            tmp2 = tmp2.reshape(1,)
            tmp3 = np.concatenate([tpl[0].ravel(), tmp1, tmp2])
            ql.append(tmp3)
        print("SAR_DATA:", sar_data[0])#sar_data[0])
        print("SAR_DATA_SHAPE:", sar_data[0].shape)
        print("SAR_DATA_TYPE:", type(sar_data[0]))
        print("SAR_DATA_LENGTH:", len(sar_data))'''
        sar_data_arr = np.array(sar_data)
        #np.savetxt("FROM_SAR_WITH_LOVE.txt", 
        #   sar_data_arr,
        #   fmt='%f')
        #print("SAR_DATA_NP_TYPE:", type(sar_data_arr))
        #print("SAR_DATA_NP_ARR:", sar_data_arr)
        return sar_data_arr

    def compute_task_embedding(self, sar_data, action_space_size):
        '''Function for computing the task embedding based on the current
        batch of SAR data derived from the replay buffer.'''
        #print("TASK NUM OF ACTIONS:", self.task.action_space.n)
        task_embedding = self.detect.lwe(sar_data, action_space_size)
        self.new_task_emb = task_embedding
        #self.current_task_emb = task_embedding
        #self.task.set_current_task_info('task_emb', task_embedding)
        #self.task.get_task()['task_emb'] = task_embedding
        self.task_emb_size = len(task_embedding)
        return task_embedding

    def calculate_emb_distance(self, current_embedding, new_embedding):
        '''Calculates the distance between the newly computed task embedding and the
        existing embedding.'''
        emb_dist = self.detect.emb_distance(current_embedding, new_embedding)
        return emb_dist

    def update_seen_tasks(self, embedding, reward, label):
        self.seen_tasks.update({self.current_task_key : {
            'task_emb': embedding,
            'reward': reward,
            'ground_truth': label
        }})
    
    def assign_task_emb(self, new_emb, emb_distance):
        '''Assigns the most up to date embedding to the current task based
        on the embedding distance threshold.

        If agent is working on the same task (< threshold) then we update embedding
        with the moving average of the new and old embeddings.

        If the agent has encountered a task shift (> threshold) then we setup a new mask
        and record in the internal registry for the new task. If it is a previously encountered
        task then we just start updating the existing record.'''

        # LEGACY CODE:
        '''# Variable used to check for key that stores the task embedding of the agent's current task
        key_to_check = 'task_emb'

        # Flag for task change detected by the detect module (True if task change happend, False otherwise).
        task_chng_bool = None

        if not key_to_check in self.task.get_task().keys():                                                 # If 'task_emb' key is not in the task_fn() registry then...
            self.task.set_current_task_info(key_to_check, torch.zeros(self.get_task_emb_size()))            # ...create the key-value pair with zeros embedding tensor
            print("TASK INFO REGISTRY UPDATED WITH EMBEDDING KEY-VALUE PAIR!!!!!!!!!!!!!!!!")

        if emb_distance < self.emb_dist_threshold:                                                          # If the computed embedding distance is below the preset fixed threshold then...
             self.task.get_task()['task_emb'] = (self.task.get_task()['task_emb'] + new_emb) / 2            # ...calculate the moving average of the new embedding and the previous average and update the task_fn() registry

             self.set_current_task_embedding(self.task.get_task()['task_emb'])                              # Update self.current_task_emb with the new averaged embedding. NOTE: We don't currently use the self.current_task_emb for anything. It's likely that this was created as a placeholder replacement of the self.current_task_label
             
             current_task_idx = self._embedding_to_idx(self.current_task_emb)                               # Gets the generated task ID if the 'new' current_task_emb is similar to any of the previously seen tasks in self.seen_tasks. Essentially we want to check if we have done this task before or not using embeddings instead of task labels. If not we get NoneType here.
             
             self.seen_tasks[current_task_idx] = self.get_current_task_embedding()                          # Update the self.seen_tasks dictionary with the self.current_task_emb (the averaged 'new' embedding)
             str_task_chng_msg = "TASK CHNAGE NNNNOOOOTTTTT DETECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

             task_chng_bool = False
             
        else:
            self.task_train_end_emb()                                                                       # End training on the previous task. This means consolidating the task mask, setting current_task_emb to NoneType and updating the number of tasks learned by the network.
            self.task_train_start_emb(new_emb)                                                              # Start training on the new task. This means checking if we have encountered the task before then set the existing model mask, otherwise, set up a new task and set the model to use a new task mask
            self.task.get_task()['task_emb'] = new_emb                                                      # Since we are switching to a different task then we update the task_fn() registry pointer with the new embedding. NOTE: The pointer is updated by the self.task.reset_task() method which sets the new task and resets the state/observations.
            self.set_current_task_embedding(new_emb)                                                        # Set the self.current_task_emb to this new task embedding. NOTE: Again this is not used for anything.
            self.set_new_task_emb(new_emb)                                                                  # Set the self.new_task_emb to this new embedding. NOTE: This is also not used for anything ???
            str_task_chng_msg = "TASK CHNAGE DETECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            
            task_chng_bool = True

        return str_task_chng_msg, task_chng_bool'''
    
        # NEW CODE:
        # Debugging CLI outputs. We track the registry for embedding updates across all tasks.

        task_change_bool = None

        # Saptarshi: Modified algorithm to use the SyncManager() internal server dictionary (self.seen_tasks) instead of the registry inside the task_fn().
        # IF SAME TASK
        if emb_distance < self.emb_dist_threshold:
            old_emb = self.seen_tasks[self.current_task_key]['task_emb']
            self.current_task_emb = (old_emb + new_emb) / 2   # Compute moving average. NOTE: We still don't use the self.current_task_emb for anything important.

            # Update dictionary at current_task_key. Ideally we would have used a pointer to get this done but unfortunately we can't do so for mp.Manager() dictionaries because it is a proxy.
            self.update_seen_tasks(
                embedding = self.current_task_emb,
                reward = np.mean(self.iteration_rewards),
                label = self.task.get_task()['task_label']
            )
            #self.seen_tasks.update({self.current_task_key : {
            #    'task_emb': self.current_task_emb,                          # Update with the moving average embedding
            #    'reward': np.mean(self.iteration_rewards),                  # Update the iteration rewards for the task
            #    'ground_truth': self.task.get_task()['task_label']          # Set the ground truth for the task. The agent does not use this but we will need it for logging and the evaluation agent.
            #}})

            task_change_bool = False

        # IF NEW TASK
        else:
            self.task_train_end_emb()                       # End training on previous task mask.
            self.task_train_start_emb(new_emb)              # Start training on new mask for the newly detected task
            self.set_current_task_embedding(new_emb)        # Set the self.current_task_emb for the newly detected task
            task_change_bool = True

        return task_change_bool

    def store_embeddings(self, new_embedding):
        '''Appends new encountered task embedding to list of encountered embeddings.'''
        self.encountered_task_embs.append(new_embedding)


    ###############################################################################
    # Methods for distilling incoming masks to model
    def distil_task_knowledge_embedding(self, mask):
        '''Method performs the distilation of task knowledge based on embeddings. The alorithmic approach
        is identical to the "distil_task_knowledge_single.'''

        # New distil task knowledge algorithm
        # this function receives only one mask
        # which is the best mask.
        #print(mask)

        #task_label = self.curr_train_task_label
        #task_idx = self._embedding_to_idx(task_embedding)

        # Process the single mask as opposed to multiple
        mask = self.vec_to_mask(mask.to(self.config.DEVICE))

        if mask is not None:
            set_mask(self.network, mask, self.current_task_key)
            return True
        else:
            return False
        
    # TODO: Saptarshi: I think we will take a task label approach to the evaluation agent
    def distil_task_knowledge_single_eval_embedding(self, mask, label):
        
        '''This metehod performs the distilation of mask based on task embedding 
        for the evaluator agent.' The aglorithmic apporach is identical to the
        "distil_task_knowledge_single" method.'''
        # New distil task knowledge algorithm
        # this function receives only one mask
        # which is the best mask.
        #print(mask)

        #task_label = self.curr_eval_task_label
        #print(task_label)
        task_idx = self._label_to_idx(label)
        #print(task_idx)

        # Process the single mask as opposed to multiple
        mask = self.vec_to_mask(mask.to(self.config.DEVICE))

        if mask is not None:
            set_mask(self.network, mask, task_idx)
            return True
        else:
            return False


    ###############################################################################
    # Getters and Setters
    def get_current_task_label(self):
        '''Get current task label'''
        return self.task.get_task()['task_label']
    def set_current_task_label(self, new_task_label):
        '''This is unlikely to be used.'''
        self.task.get_task()['task_label'] = new_task_label

    def get_task_action_space_size(self):
        '''A getter method for retreiving the task action space.'''
        return self.task_action_space_size
    def set_task_action_space_size(self, new_action_space):
        '''A setter method for manually assining the task action space
        size for letter use it for converting the actions to one-hot
        representation.'''
        self.task_action_space_size = new_action_space
    
    def get_task_emb_size(self):
        '''''A getter method for retreiving the task embedding size.'''
        return self.task_emb_size
    def set_task_emb_size(self, new_embedding_size):
        '''A setter for dynamically setting the mebedding size.'''
        self.task_emb_size = new_embedding_size
    
    def get_emb_dist_threshold(self):
        '''A getter method for reteiving the distnce threshold for the embeddings.'''
        return self.emb_dist_threshold
    def set_emb_dist_threshold(self, new_distance_threshold):
        '''A setter for dynamically setting the distance threshold of the embeddings.'''
        self.emb_dist_threshold = new_distance_threshold
    
    def get_detect_module_activation_frequency(self):
        '''A getteer for the detect module activation frequency'''
        return self.detect_module_activation_frequency
    def set_detect_module_activation_frequency(self, new_activation_frequency):
        '''A setter method for manually setting the detect module activation frequnecy.'''
        self.detect_module_activation_frequncy = new_activation_frequency
    
    def get_current_task_embedding(self):
        '''A getter method for retreiving the current task embedding.'''
        return self.current_task_emb
    def set_current_task_embedding(self, new_current_task_embedding):
        '''A setter method for updating the current task embedding with the new
        more acurate estimate of the embedding produced by the detect module.'''
        self.current_task_emb = new_current_task_embedding
    
    def get_new_task_embedding(self):
        ''''''
        return self.new_task_emb
    def set_new_task_emb(self, new_emb):
        ''''''
        self.new_task_emb = new_emb

