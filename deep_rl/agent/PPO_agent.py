#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from copy import deepcopy
import numpy as np

class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            actions, log_probs, _, values = self.network.predict(states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states

        self.states = states
        pending_value = self.network.predict(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs, entropy_loss, values = self.network.predict(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps

class PPOContinualLearnerAgent(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)
        self.config = config
        self.task = None if config.task_fn is None else config.task_fn()
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task
        tasks = self.task.get_all_tasks(config.cl_requires_task_label)[ : config.cl_num_tasks]
        self.config.cl_tasks_info = tasks
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])

        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        self.opt = config.optimizer_fn(self.network.parameters(), config.lr)
        self.total_steps = 0
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.layers_output = None # for logging per layer output and sometimes l1/l2 regularisation

        self.data_buffer = []
        # weight preservation setup
        self.params = {n: p for n, p in self.network.named_parameters() if p.requires_grad}
        self.precision_matrices = {}
        self.means = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.precision_matrices[n] = p.data.to(config.DEVICE)
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.means[n] = p.data.to(config.DEVICE)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-2]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, return_layer_output=True)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                weight_pres_loss = self.penalty()

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss).backward()
                #(policy_loss + value_loss).backward() # NOTE restore this please
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def penalty(self):
        loss = 0
        for n, p in self.network.named_parameters():
            _loss = self.precision_matrices[n] * (p - self.means[n]) ** 2
            loss += _loss.sum()
        return loss * self.config.cl_loss_coeff

    def consolidate(self, batch_size=32):
        raise NotImplementedError

class PPOAgentBaseline(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent without preservation/consolidation
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=32):
        print('sanity check, calling consolidate in agent w/o preservation')
        # the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices

class PPOAgentBaselineL1Weights(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent without preservation/consolidation
    sparsify weights of the network via L1 regularisation on the weight values
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-2]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, return_layer_output=True)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                #weight_pres_loss = self.penalty()
                weight_pres_loss = 0.

                # regularisation loss (l1)
                params_ = nn.utils.parameters_to_vector(self.network.parameters())
                reg_loss = config.reg_loss_coeff * (torch.norm(params_, p=1))

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss + reg_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def consolidate(self, batch_size=32):
        print('sanity check, calling consolidate in agent w/o preservation (sparse weights by L1)')
        # the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices

class PPOAgentBaselineL2Weights(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent without preservation/consolidation.
    sparsify weights of the network via L2 regularisation on the weight values
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-2]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, return_layer_output=True)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                #weight_pres_loss = self.penalty()
                weight_pres_loss = 0.

                # regularisation loss (l2)
                params_ = nn.utils.parameters_to_vector(self.network.parameters())
                reg_loss = config.reg_loss_coeff * (torch.norm(params_, p=2) ** 2)

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss + reg_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def consolidate(self, batch_size=32):
        print('sanity check, calling consolidate in agent w/o preservation (sparse weights by L2)')
        # the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices

class PPOAgentBaselineGroupL1Weights(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent without preservation/consolidation
    sparsify weights of the network via Group L1 (Lasso) regularisation on the weight values
    https://arxiv.org/abs/1607.00485
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-2]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, return_layer_output=True)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                #weight_pres_loss = self.penalty()
                weight_pres_loss = 0.

                # regularisation group l1 (i.e., group lasso)
                reg_loss = 0.
                for param in self.network.parameters():
                    reg_loss += torch.sum(np.sqrt(param.shape[0]) * torch.norm(param, p=2, dim=0))
                reg_loss = config.reg_loss_coeff * reg_loss

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss + reg_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def consolidate(self, batch_size=32):
        print('sanity check, calling consolidate in agent w/o preservation (sparse weights by group L1)')
        # the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices

class PPOAgentBaselineSparseGroupL1Weights(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent without preservation/consolidation
    sparsify weights of the network via Sparse Group L1 (Lasso) regularisation on the weight values
    https://arxiv.org/abs/1607.00485
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-2]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, return_layer_output=True)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                #weight_pres_loss = self.penalty()
                weight_pres_loss = 0.

                # regularisation sparse group l1 (group l1 + l1)
                # first, group l1 regularisation
                g_reg_loss = 0.
                for param in self.network.parameters():
                    g_reg_loss += torch.sum(np.sqrt(param.shape[0]) * torch.norm(param, p=2, dim=0))
                # second, l1 regularisation
                params_ = nn.utils.parameters_to_vector(self.network.parameters())
                l1_reg_loss = torch.norm(params_, p=1)
                # final step
                reg_loss = config.reg_loss_coeff * (g_reg_loss + l1_reg_loss)

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss + reg_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def consolidate(self, batch_size=32):
        print('sanity check, calling consolidate in agent w/o preservation (sparse weights by sparse group L1)')
        # the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices

class PPOAgentBaselineL1Act(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent without preservation/consolidation
    sparsify weights of the network via L1 regularisation on activations per layer
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)
        self.layers_output = None

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-2]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, return_layer_output=True)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                #weight_pres_loss = self.penalty()
                weight_pres_loss = 0.

                # per layer output regularisation loss
                out_reg_loss = 0.
                for _, out in outs:
                    out_reg_loss += torch.norm(out, p=1, dim=1).mean()
                out_reg_loss *= config.reg_loss_coeff

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss + out_reg_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def consolidate(self, batch_size=32):
        print('sanity check,calling consolidate in agent w/o preservation(sparse activation by L1)')
        # the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices

class PPOAgentBaselineL2Act(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent without preservation/consolidation.
    sparsify weights of the network via L2 regularisation on activations per layer
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)
        self.layers_output = None

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-2]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, return_layer_output=True)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                #weight_pres_loss = self.penalty()
                weight_pres_loss = 0.

                # per layer output regularisation loss
                out_reg_loss = 0.
                for _, out in outs:
                    out_reg_loss += torch.norm(out, p=2, dim=1).mean()
                out_reg_loss *= config.reg_loss_coeff

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss + out_reg_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def consolidate(self, batch_size=32):
        print('sanity check,calling consolidate in agent w/o preservation(sparse activation by L2)')
        # the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices

class PPOAgentSCPwithModulatedGradients(PPOContinualLearnerAgent):
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

        # neuromodulated (hypernet) mask for modulated gradients
        tasks = self.config.cl_tasks_info
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])
        self.nm_mask = {}
        for n, p in self.network.named_parameters():
            self.nm_mask[n] = torch.zeros_like(p).to(self.config.DEVICE)
        self.nm_nets = {}
        for n, p in self.network.named_parameters():
            m_ = NMNet(np.array(p.shape)[::-1], label_dim)
            m_.to(config.DEVICE)
            self.nm_nets[n] = m_

    def iteration(self):
        # NOTE generate mask
        self._generate_nm_mask()

        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-1]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                weight_pres_loss = self.penalty()

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss).backward()

                # NOTE apply mask to gradients to ensure that only certain parameters can be updated
                for n, p in self.network.named_parameters():
                    p.grad = p.grad * self.nm_mask[n].reshape(*p.shape)

                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        return np.mean(grad_norms_).mean()

    def _generate_nm_mask(self):
        task_label = self.task.get_task()['task_label'] # current task
        for n, nm_net in self.nm_nets.items():
            self.nm_mask[n] = nm_net(task_label.reshape(1, -1))

    def penalty(self):
        loss = 0
        for n, p in self.network.named_parameters():
            _loss = self.precision_matrices[n] * (p - self.means[n]) ** 2
            loss += _loss.sum()
        return loss * self.config.cl_loss_coeff

    def consolidate(self, batch_size=32):
        print('sanity check, in scp consolidation with modulated gradients')
        states = np.concatenate(self.data_buffer)
        states = states[-512 : ] # NOTE, hack downsizing buffer. please remove later
        task_label = self.task.get_task()['task_label']
        task_label = np.repeat(task_label.reshape(1, -1), len(states), axis=0)
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        # Get network outputs
        idxs = np.arange(len(states))
        np.random.shuffle(idxs)
        # shuffle data
        states = states[idxs]
        task_label = task_label[idxs]

        num_batches = len(states) // batch_size
        num_batches = num_batches + 1 if len(states) % batch_size > 0 else num_batches
        for batch_idx in range(num_batches):
            start, end = batch_idx * batch_size, (batch_idx+1) * batch_size
            states_ = states[start:end, ...]
            task_label_ = task_label[start:end, ...]
            self.network.zero_grad()
            logits, actions, _, _, values = self.network.predict(states_, task_label=task_label_)
            # actor consolidation
            logits_mean = logits.type(torch.float32).mean(dim=0)
            K = logits_mean.shape[0]
            for _ in range(config.cl_n_slices):
                xi = torch.randn(K, ).to(config.DEVICE)
                xi /= torch.sqrt((xi**2).sum())
                self.network.zero_grad()
                out = torch.matmul(logits_mean, xi)
                out.backward(retain_graph=True)
                # NOTE apply mask to gradients to ensure that only certain parameters can be updated
                for n, p in self.params.items():
                    # detach mask from computation graph
                    p.grad = p.grad * self.nm_mask[n].detach().reshape(*p.shape) 
                # Update the temporary precision matrix
                for n, p in self.params.items():
                    precision_matrices[n].data += p.grad.data ** 2
            # critic consolidation
            values_mean = values.type(torch.float32).mean(dim=0)
            self.network.zero_grad()
            values_mean.backward(retain_graph=True)
            # NOTE apply mask to gradients to ensure that only certain parameters can be updated
            for n, p in self.params.items():
                # detach mask from computation graph
                p.grad = p.grad * self.nm_mask[n].detach().reshape(*p.shape) 
            # Update the temporary precision matrix
            for n, p in self.params.items():
                precision_matrices[n].data += p.grad.data ** 2

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(config.DEVICE)

        self.network.train()
        # return task precision matrices and general precision matrices across tasks agent has
        # been explosed to so far
        return precision_matrices, self.precision_matrices

class PPOAgentSCPwithModulatedImportances(PPOContinualLearnerAgent):
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

        # neuromodulated (hypernet) mask for modulated importances
        tasks = self.config.cl_tasks_info
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])
        self.nm_mask = {}
        for n, p in self.network.named_parameters():
            self.nm_mask[n] = torch.zeros_like(p).to(self.config.DEVICE)
        self.nm_nets = {}
        for n, p in self.network.named_parameters():
            m_ = NMNet(np.array(p.shape)[::-1], label_dim)
            m_.to(config.DEVICE)
            self.nm_nets[n] = m_

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-1]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                weight_pres_loss = self.penalty()

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        return np.mean(grad_norms_).mean()

    def _generate_nm_mask(self):
        task_label = self.task.get_task()['task_label'] # current task
        for n, nm_net in self.nm_nets.items():
            self.nm_mask[n] = nm_net(task_label.reshape(1, -1))

    def penalty(self):
        self._generate_nm_mask()
        loss = 0
        for n, p in self.network.named_parameters():
            #_loss = self.precision_matrices[n] * (p - self.means[n]) ** 2
            # NOTE neuromodulated importances
            _nm = self.nm_mask[n].reshape(*p.shape)
            _loss = _nm * self.precision_matrices[n] * (p - self.means[n]) ** 2
            loss += _loss.sum()
        return loss * self.config.cl_loss_coeff

    def consolidate(self, batch_size=32):
        print('sanity check, in scp consolidation with modulated importances')
        states = np.concatenate(self.data_buffer)
        states = states[-512 : ] # NOTE, hack downsizing buffer. please remove later
        task_label = self.task.get_task()['task_label']
        task_label = np.repeat(task_label.reshape(1, -1), len(states), axis=0)
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        # Get network outputs
        idxs = np.arange(len(states))
        np.random.shuffle(idxs)
        # shuffle data
        states = states[idxs]
        task_label = task_label[idxs]

        num_batches = len(states) // batch_size
        num_batches = num_batches + 1 if len(states) % batch_size > 0 else num_batches
        for batch_idx in range(num_batches):
            start, end = batch_idx * batch_size, (batch_idx+1) * batch_size
            states_ = states[start:end, ...]
            task_label_ = task_label[start:end, ...]
            self.network.zero_grad()
            logits, actions, _, _, values = self.network.predict(states_, task_label=task_label_)
            # actor consolidation
            logits_mean = logits.type(torch.float32).mean(dim=0)
            K = logits_mean.shape[0]
            for _ in range(config.cl_n_slices):
                xi = torch.randn(K, ).to(config.DEVICE)
                xi /= torch.sqrt((xi**2).sum())
                self.network.zero_grad()
                out = torch.matmul(logits_mean, xi)
                out.backward(retain_graph=True)
                # Update the temporary precision matrix
                for n, p in self.params.items():
                    precision_matrices[n].data += p.grad.data ** 2
            # critic consolidation
            values_mean = values.type(torch.float32).mean(dim=0)
            self.network.zero_grad()
            values_mean.backward(retain_graph=True)
            # Update the temporary precision matrix
            for n, p in self.params.items():
                precision_matrices[n].data += p.grad.data ** 2

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(config.DEVICE)

        self.network.train()
        # return task precision matrices and general precision matrices across tasks agent has
        # been explosed to so far
        return precision_matrices, self.precision_matrices

class PPOAgentSCPwithMasking(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using sliced cramer preservation (SCP)
    weight preservation mechanism
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def iteration(self, mask):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label'] # current task
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)

        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-1]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        grad_norms_ = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                _, _, log_probs, entropy_loss, values = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                weight_pres_loss = self.penalty()

                self.opt.zero_grad()
                (policy_loss + value_loss + weight_pres_loss).backward()

                # NOTE apply mask to gradients to ensure that only certain parameters can be updated
                for n, p in self.network.named_parameters():
                    p.grad = p.grad * mask[n]

                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        return np.mean(grad_norms_).mean()

    #def consolidate(self, batch_size=32):
    def consolidate(self, mask, batch_size=32):
        print('sanity check, in scp consolidation with masking')
        states = np.concatenate(self.data_buffer)
        states = states[-512 : ] # NOTE, hack downsizing buffer. please remove later
        task_label = self.task.get_task()['task_label']
        task_label = np.repeat(task_label.reshape(1, -1), len(states), axis=0)
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        # Get network outputs
        idxs = np.arange(len(states))
        np.random.shuffle(idxs)
        # shuffle data
        states = states[idxs]
        task_label = task_label[idxs]

        num_batches = len(states) // batch_size
        num_batches = num_batches + 1 if len(states) % batch_size > 0 else num_batches
        for batch_idx in range(num_batches):
            start, end = batch_idx * batch_size, (batch_idx+1) * batch_size
            states_ = states[start:end, ...]
            task_label_ = task_label[start:end, ...]
            self.network.zero_grad()
            logits, actions, _, _, values = self.network.predict(states_, task_label=task_label_)
            # actor consolidation
            logits_mean = logits.type(torch.float32).mean(dim=0)
            K = logits_mean.shape[0]
            for _ in range(config.cl_n_slices):
                xi = torch.randn(K, ).to(config.DEVICE)
                xi /= torch.sqrt((xi**2).sum())
                self.network.zero_grad()
                out = torch.matmul(logits_mean, xi)
                out.backward(retain_graph=True)

                # NOTE apply mask to gradients to ensure that only certain parameters can be updated
                for n, p in self.params.items():
                    p.grad = p.grad * mask[n]
                # Update the temporary precision matrix
                for n, p in self.params.items():
                    precision_matrices[n].data += p.grad.data ** 2

            # critic consolidation
            values_mean = values.type(torch.float32).mean(dim=0)
            self.network.zero_grad()
            values_mean.backward(retain_graph=True)

            # NOTE apply mask to gradients to ensure that only certain parameters can be updated
            for n, p in self.params.items():
                p.grad = p.grad * mask[n]
            # Update the temporary precision matrix
            for n, p in self.params.items():
                precision_matrices[n].data += p.grad.data ** 2

        # clamp vavlues in precision matrices. values below a min threshold is set to 0
        # while values above a max threshold is set to the max threshold value
        for n, p in self.network.named_parameters():
            precision_matrices[n][precision_matrices[n] < config.cl_pm_min] = 0.
            precision_matrices[n][precision_matrices[n] > config.cl_pm_max] = config.cl_pm_max

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(config.DEVICE)

        self.network.train()
        # return task precision matrices and general precision matrices across tasks agent has
        # been explosed to so far
        return precision_matrices, self.precision_matrices


class PPOAgentMAS(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using memory aware synapse (MAS)
    weight preservation mechanism
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=32):
        states = np.concatenate(self.data_buffer)
        task_label = self.task.get_task()['task_label']
        task_label = np.repeat(task_label.reshape(1, -1), len(states), axis=0)

        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        # shuffle data
        idxs = np.arange(len(states))
        np.random.shuffle(idxs)
        states = states[idxs]
        task_label = task_label[idxs]

        num_batches = len(states) // batch_size
        num_batches = num_batches + 1 if len(states) % batch_size > 0 else num_batches
        for batch_idx in range(num_batches):
            start, end = batch_idx * batch_size, (batch_idx+1) * batch_size
            states_ = states[start:end, ...]
            task_label_ = task_label[start:end, ...]
            logits, actions, _, _, values = self.network.predict(states_, task_label=task_label_)
            logits = torch.softmax(logits, dim=1)
            # get value loss
            value_loss = values.sum()
            # get actor loss: l2 norm of each row of logits
            try:
                actor_loss = (torch.linalg.norm(logits, ord=2, dim=1)).sum()
            except:
                # older version of pytorch, we calculate l2 norm as API is not available
                actor_loss = (logits ** 2).sum(dim=1).sqrt().sum()

            self.network.zero_grad()
            actor_loss.backward()
            # Update the temporary precision matrix
            for n, p in self.params.items():
                #precision_matrices[n].data += p.grad.data ** 2 / float(len(states))
                precision_matrices[n].data += p.grad.data ** 2
            self.network.zero_grad()
            value_loss.backward()
            # Update the temporary precision matrix
            for n, p in self.params.items():
                #precision_matrices[n].data += p.grad.data ** 2 / float(len(states))
                precision_matrices[n].data += p.grad.data ** 2

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(config.DEVICE)

        self.network.train()
        # return task precision matrices and general precision matrices across tasks agent has
        # been explosed to so far
        return precision_matrices, self.precision_matrices

class PPOAgentSCP(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using sliced cramer preservation (SCP)
    weight preservation mechanism
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=32):
        print('sanity check, in scp consolidation')
        states = np.concatenate(self.data_buffer)
        states = states[-512 : ] # NOTE, hack downsizing buffer. please remove later
        task_label = self.task.get_task()['task_label']
        task_label = np.repeat(task_label.reshape(1, -1), len(states), axis=0)
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        # Get network outputs
        idxs = np.arange(len(states))
        np.random.shuffle(idxs)
        # shuffle data
        states = states[idxs]
        task_label = task_label[idxs]

        num_batches = len(states) // batch_size
        num_batches = num_batches + 1 if len(states) % batch_size > 0 else num_batches
        for batch_idx in range(num_batches):
            start, end = batch_idx * batch_size, (batch_idx+1) * batch_size
            states_ = states[start:end, ...]
            task_label_ = task_label[start:end, ...]
            self.network.zero_grad()
            logits, actions, _, _, values = self.network.predict(states_, task_label=task_label_)
            # actor consolidation
            logits_mean = logits.type(torch.float32).mean(dim=0)
            K = logits_mean.shape[0]
            for _ in range(config.cl_n_slices):
                xi = torch.randn(K, ).to(config.DEVICE)
                xi /= torch.sqrt((xi**2).sum())
                self.network.zero_grad()
                out = torch.matmul(logits_mean, xi)
                out.backward(retain_graph=True)
                # Update the temporary precision matrix
                for n, p in self.params.items():
                    precision_matrices[n].data += p.grad.data ** 2
            # critic consolidation
            values_mean = values.type(torch.float32).mean(dim=0)
            self.network.zero_grad()
            values_mean.backward(retain_graph=True)
            # Update the temporary precision matrix
            for n, p in self.params.items():
                precision_matrices[n].data += p.grad.data ** 2

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(config.DEVICE)

        self.network.train()
        # return task precision matrices and general precision matrices across tasks agent has
        # been explosed to so far
        return precision_matrices, self.precision_matrices


class PPOAgentEWC(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using elastic weight consolidation (EWC)
    weight preservation mechanism
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=16):
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()
        # collect data and consolidate
        self.states = config.state_normalizer(self.task.reset())

        for _ in range(batch_size):
            rollout = []
            states = self.states

            #current_task_label = self.task.get_task()['task_label']
            #batch_dim = len(states) # same as config.num_workers
            #if batch_dim == 1:
            #    current_task_label = current_task_label.reshape(1, -1)
            #else:
            #    current_task_label = np.repeat(current_task_label.reshape(1, -1), batch_dim, axis=0)
            task_label = self.task.get_task()['task_label'] # current task
            batch_dim = len(states) # same as config.num_workers
            if batch_dim == 1:
                batch_task_label = task_label.reshape(1, -1)
            else:
                batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)


            for _ in range(config.rollout_length):
                _, actions, log_probs, _, values = self.network.predict(states, \
                    task_label=batch_task_label)
                next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
                self.episode_rewards += rewards
                rewards = config.reward_normalizer(rewards)
                for i, terminal in enumerate(terminals):
                    if terminals[i]:
                        self.last_episode_rewards[i] = self.episode_rewards[i]
                        self.episode_rewards[i] = 0
                next_states = config.state_normalizer(next_states)
                rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                    rewards, 1 - terminals])
                states = next_states

            self.states = states
            pending_value = self.network.predict(states, task_label=batch_task_label)[-1]
            rollout.append([states, pending_value, None, None, None, None])
            processed_rollout = [None] * (len(rollout) - 1)
            advantages = tensor(np.zeros((config.num_workers, 1)))
            returns = pending_value.detach()
            for i in reversed(range(len(rollout) - 1)):
                states, value, actions, log_probs, rewards, terminals = rollout[i]
                terminals = tensor(terminals).unsqueeze(1)
                rewards = tensor(rewards).unsqueeze(1)
                actions = tensor(actions)
                states = tensor(states)
                next_value = rollout[i + 1][1]
                returns = rewards + config.discount * terminals * returns
                if not config.use_gae:
                    advantages = returns - value.detach()
                else:
                    td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                    advantages = advantages * config.gae_tau * config.discount * terminals + td_error
                processed_rollout[i] = [states, actions, log_probs, returns, advantages]

            states, actions, log_probs_old, returns, advantages=map(lambda x: torch.cat(x, dim=0), \
                zip(*processed_rollout))
            advantages = (advantages - advantages.mean()) / advantages.std()

            batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
            for _ in range(config.optimization_epochs):
                batcher.shuffle()
                while not batcher.end():
                    batch_indices = batcher.next_batch()[0]
                    batch_indices = tensor(batch_indices).long()
                    sampled_states = states[batch_indices]
                    sampled_actions = actions[batch_indices]
                    sampled_log_probs_old = log_probs_old[batch_indices]
                    sampled_returns = returns[batch_indices]
                    sampled_advantages = advantages[batch_indices]

                    #_, log_probs, entropy_loss, values = self.network.predict(sampled_states, \
                    #    sampled_actions)
                    batch_dim = sampled_states.shape[0]
                    batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                    _, _, log_probs, entropy_loss, values = self.network.predict(sampled_states, \
                        sampled_actions, task_label=batch_task_label)
                    ratio = (log_probs - sampled_log_probs_old).exp()
                    obj = ratio * sampled_advantages
                    obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                              1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                    policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                        - config.entropy_weight * entropy_loss.mean()

                    value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                    self.network.zero_grad()
                    loss = policy_loss + value_loss
                    loss.backward()

                    # Update the temporary precision matrix
                    for n, p in self.params.items():
                        #precision_matrices[n].data += p.grad.data ** 2 / float(len(sampled_states))
                        precision_matrices[n].data += p.grad.data ** 2

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(config.DEVICE)

        self.network.train()
        # return task precision matrices and general precision matrices across tasks agent has
        # been explosed to so far
        return precision_matrices, self.precision_matrices
