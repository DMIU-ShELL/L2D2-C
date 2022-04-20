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

        self.task_label_param = torch.zeros((label_dim,)).to(config.DEVICE)
        self.task_label_param.requires_grad = True
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        if config.cl_learn_task_label:
            _params = list(self.network.parameters()) + [self.task_label_param,]
        else:
            _params = list(self.network.parameters())
        self.opt = config.optimizer_fn(_params, config.lr)
        self.total_steps = 0
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.layers_output = None # for logging per layer output and sometimes l1/l2 regularisation

        #self.data_buffer = []
        self.data_buffer = Replay(memory_size=int(3e4), batch_size=1024)
        #self.data_buffer = Replay(memory_size=int(256), batch_size=32) # for dynamic grid
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
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
                grad_norms_.append(norm_.detach().cpu().numpy())
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_)

    def penalty(self):
        loss = 0
        for n, p in self.network.named_parameters():
            _loss = self.precision_matrices[n] * (p - self.means[n]) ** 2
            loss += _loss.sum()
        return loss * self.config.cl_loss_coeff

    def consolidate(self, batch_size=32):
        raise NotImplementedError

    def task_train_start(self):
        config = self.config
        task_info = self.task.get_task()
        # task_label_agent should be available by the second time the task 
        # if encountered by the agent
        if 'task_label_agent' in task_info.keys(): 
            self.task_label_param.data = tensor(task_info['task_label_agent'])
        else:
            self.task_label_param.data = tensor(task_info['task_label'])

    def task_train_end(self):
        config = self.config
        task_info = self.task.get_task()
        #task_info['task_label_agent'] = self.task_label_param.detach().cpu().numpy()
        #print(task_info)
        #print(self.task.get_task())
        #print('\n\n\n\n')
        #return self.consolidate()
        return {'task_label_agent': self.task_label_param.detach().cpu().numpy(), \
            'consolidate': self.consolidate()}

class PPOAgentBaseline(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent without preservation/consolidation
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=32):
        config = self.config
        config.logger.info('sanity check, calling consolidate in agent w/o preservation')
        # the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices

##################################################################################
##
## PPO + NO CONSOLIDATION + SPARSITY
##
##################################################################################
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
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
                grad_norms_.append(norm_.detach().cpu().numpy())
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
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
                grad_norms_.append(norm_.detach().cpu().numpy())
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
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
                grad_norms_.append(norm_.detach().cpu().numpy())
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
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
                grad_norms_.append(norm_.detach().cpu().numpy())
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
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
                grad_norms_.append(norm_.detach().cpu().numpy())
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
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
                grad_norms_.append(norm_.detach().cpu().numpy())
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

##################################################################################
##
## PPO + CONSOLIDATION + SPARSITY
##
##################################################################################
class PPOAgentSCPSparseGroupL1Weights(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent with scp preservation/consolidation
    sparsify weights of the network via Sparse Group L1 (Lasso) regularisation on the weight values
    https://arxiv.org/abs/1607.00485
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
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
                grad_norms_.append(norm_.detach().cpu().numpy())
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def consolidate(self, batch_size=32):
        print('sanity check, consolidate in ppo + scp + sparse-group-l1 reg agent')
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        #task_label = self.task.get_task()['task_label']
        #task_label_ = np.repeat(task_label.reshape(1, -1), batch_size, axis=0)
        task_label = self.task_label_param.detach()
        task_label_ = torch.repeat_interleave(task_label.reshape(1, -1), batch_size, dim=0)
        num_batches = self.data_buffer.size() // batch_size
        for batch_idx in range(num_batches):
            states_, = self.data_buffer.sample(batch_size)
            self.network.zero_grad()
            logits, actions, _, _, values, _ = self.network.predict(states_, task_label=task_label_)
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


##################################################################################
##
## PPO + CONSOLIDATION
##
##################################################################################
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
            #task_label = self.task.get_task()['task_label'] # current task
            task_label = self.task_label_param.detach()
            batch_dim = len(states) # same as config.num_workers
            if batch_dim == 1:
                batch_task_label = task_label.reshape(1, -1)
            else:
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)


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


class PPOAgentMAS(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using memory aware synapse (MAS)
    weight preservation mechanism
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=32):
        print('sanity check, in mas consolidation')
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        #task_label = self.task.get_task()['task_label']
        #task_label_ = np.repeat(task_label.reshape(1, -1), batch_size, axis=0)
        task_label = self.task_label_param.detach()
        task_label_ = torch.repeat_interleave(task_label.reshape(1, -1), batch_size, dim=0)
        num_batches = self.data_buffer.size() // batch_size
        for batch_idx in range(num_batches):
            states_, = self.data_buffer.sample(batch_size)
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
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        #task_label = self.task.get_task()['task_label']
        #task_label_ = np.repeat(task_label.reshape(1, -1), batch_size, axis=0)
        task_label = self.task_label_param.detach()
        task_label_ = torch.repeat_interleave(task_label.reshape(1, -1), batch_size, dim=0)
        num_batches = self.data_buffer.size() // batch_size
        for batch_idx in range(num_batches):
            states_, = self.data_buffer.sample(batch_size)
            self.network.zero_grad()
            logits, actions, _, _, values, _ = self.network.predict(states_, task_label=task_label_)
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

##################################################################################
##
## PPO + CONSOLIDATION + Modulation / Masking
##
##################################################################################
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
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
                grad_norms_.append(norm_.detach().cpu().numpy())
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
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
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        #task_label = self.task.get_task()['task_label']
        #task_label_ = np.repeat(task_label.reshape(1, -1), batch_size, axis=0)
        task_label = self.task_label_param.detach()
        task_label_ = torch.repeat_interleave(task_label.reshape(1, -1), batch_size, dim=0)
        num_batches = self.data_buffer.size() // batch_size
        for batch_idx in range(num_batches):
            states_, = self.data_buffer.sample(batch_size)
            self.network.zero_grad()
            logits, actions, _, _, values, _ = self.network.predict(states_, task_label=task_label_)
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
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
                grad_norms_.append(norm_.detach().cpu().numpy())
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
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
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        #task_label = self.task.get_task()['task_label']
        #task_label_ = np.repeat(task_label.reshape(1, -1), batch_size, axis=0)
        task_label = self.task_label_param.detach()
        task_label_ = torch.repeat_interleave(task_label.reshape(1, -1), batch_size, dim=0)
        num_batches = self.data_buffer.size() // batch_size
        for batch_idx in range(num_batches):
            states_, = self.data_buffer.sample(batch_size)
            self.network.zero_grad()
            logits, actions, _, _, values, _ = self.network.predict(states_, task_label=task_label_)
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
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
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
                grad_norms_.append(norm_.detach().cpu().numpy())
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    #def consolidate(self, batch_size=32):
    def consolidate(self, mask, batch_size=32):
        print('sanity check, in scp consolidation with masking')
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        #task_label = self.task.get_task()['task_label']
        #task_label_ = np.repeat(task_label.reshape(1, -1), batch_size, axis=0)
        task_label = self.task_label_param.detach()
        task_label_ = torch.repeat_interleave(task_label.reshape(1, -1), batch_size, dim=0)
        num_batches = self.data_buffer.size() // batch_size
        for batch_idx in range(num_batches):
            states_, = self.data_buffer.sample(batch_size)
            self.network.zero_grad()
            logits, actions, _, _, values, _ = self.network.predict(states_, task_label=task_label_)
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

class PPOAgentSCPModulatedFP(PPOContinualLearnerAgent):
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

        # neuromodulated (hypernet) mask for modulated forward pass (FP)
        tasks = self.config.cl_tasks_info
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])
        self.nm_mask = {}
        self.mask_real = {}
        for n, p in self.network.named_parameters():
            self.nm_mask[n] = torch.zeros_like(p).to(self.config.DEVICE)
            self.mask_real[n] = torch.zeros_like(p).to(self.config.DEVICE)
        self.nm_nets = {}
        for n, p in self.network.named_parameters():
            #m_ = NMNet(np.array(p.shape)[::-1], label_dim)
            #m_ = NMNetKWinners(np.array(p.shape)[::-1], label_dim)
            m_ = NMNetKWinners(np.array(p.shape)[::-1], label_dim, 0.1)
            m_.to(config.DEVICE)
            self.nm_nets[n] = m_

        # pm of nm nets
        self.nm_pm = {}
        self.nm_means = {}
        for k, net in self.nm_nets.items():
            self.nm_pm[k] = {}
            self.nm_means[k] = {}
            for n, p in net.named_parameters():
                p = deepcopy(p)
                p.data.zero_()
                self.nm_pm[k][n] = p.data.to(config.DEVICE)
            for n, p in net.named_parameters():
                p = deepcopy(p)
                p.data.zero_()
                self.nm_means[k][n] = p.data.to(config.DEVICE)

        all_parameters = []
        all_parameters += list(self.network.parameters())
        for _, nm_model in self.nm_nets.items():
            all_parameters += list(nm_model.parameters())
        self.opt = config.optimizer_fn(all_parameters, config.lr)

    def iteration(self):
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        # NOTE generate mask
        self._generate_nm_mask(task_label)

        config = self.config
        rollout = []
        states = self.states
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label, masks=self.nm_mask)
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label, masks=self.nm_mask)[-2]
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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, masks=self.nm_mask)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                weight_pres_loss = self.penalty()

                self.opt.zero_grad()

                # loss continuous mask
                (policy_loss + value_loss + weight_pres_loss).backward()
                # loss binary mask
                #(policy_loss + value_loss + weight_pres_loss).backward(retain_graph=True)
                #for n in self.nm_mask.keys():
                #    mb = self.nm_mask[n]
                #    mr = self.mask_real[n]
                #    if mr.grad is not None:
                #        #print(n)
                #        #print('mr.shape:', mr.shape, 'mr.grad:')
                #        #print(mr.grad)
                #        #print('mb.shape:', mb.shape, 'mb.grad:')
                #        #print(mb.grad)
                #        mr.backward(mb.grad)

                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_.detach().cpu().numpy())
                self.opt.step()

                #for n, m in self.nm_nets.items():
                #    print(n)
                #    for p in m.parameters():
                #        print(p.grad)
                #import sys; sys.exit();

                # NOTE generate mask for next batch of training
                self._generate_nm_mask(task_label)

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def _generate_nm_mask(self, task_label):
        for n, nm_net in self.nm_nets.items():
            mr = nm_net(task_label.reshape(1, -1))
            if mr.requires_grad is True: 
                mr.retain_grad()
            self.mask_real[n] = mr
            # m1: binarize
            #mb = torch.sign(mr)
            # m2: continuous (sigmoided)
            #mb = torch.zeros_like(mr)
            #mb[mr > 0.] = torch.sigmoid(mr[mr > 0.] + 5.)
            # m3: continuous raw
            mb = mr

            if mb.requires_grad is True:
                mb.retain_grad()
            self.nm_mask[n] = mb

    def penalty(self):
        losses = []
        for k, net in self.nm_nets.items():
            loss = 0
            for n, p in net.named_parameters():
                _loss = self.nm_pm[k][n] * (p - self.nm_means[k][n]) ** 2
                loss += _loss.sum()
            loss = loss * self.config.cl_loss_coeff
            losses.append(loss)
        losses = torch.stack(losses)
        return losses.sum()

    def consolidate(self, batch_size=32):
        print('sanity check, in scp consolidation with modulated fp')
        config = self.config
        
        #task_label = self.task.get_task()['task_label'] # current task
        #task_label = task_label.reshape(1, -1)
        task_label = self.task_label_param.detach()
        #task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_size, dim=0)
        task_label = task_label.reshape(1, -1)
        for k, net in self.nm_nets.items():
            pm = {}
            for n, p in net.named_parameters():
                pm[n] = deepcopy(p)
                pm[n].data.zero_()
                pm[n] = pm[n].data.to(config.DEVICE)
            net.zero_grad()
            outs = net(task_label) # forward pass
            outs = outs.view(1, -1)
            outs_mean = outs.type(torch.float32).mean(dim=0)
            K = outs_mean.shape[0]
            for _ in range(config.cl_n_slices):
                xi = torch.randn(K, ).to(config.DEVICE)
                xi /= torch.sqrt((xi**2).sum())
                net.zero_grad()
                loss = torch.matmul(outs_mean, xi)
                loss.backward(retain_graph=True)
                # Update the temporary precision matrix
                for n, p in net.named_parameters():
                    pm[n].data += p.grad.data ** 2

            for n, p in net.named_parameters():
                if p.requires_grad is False: continue
                # Update the precision matrix
                self.nm_pm[k][n] = config.cl_alpha*self.nm_pm[k][n] + (1 - config.cl_alpha) * pm[n]
                # Update the means
                self.nm_means[k][n] = deepcopy(p.data).to(config.DEVICE)

        # this return values are not useful for the class. only added for compatibility purpose.
        # consolidation is not applied to the target network.
        return self.precision_matrices, self.precision_matrices

    def evaluation_action(self, state, task_label):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        task_label = np.stack([task_label])
        out = self.network.predict(state, task_label=task_label, masks=self.nm_mask)
        self.config.state_normalizer.unset_read_only()
        if isinstance(out, dict) or isinstance(out, list) or isinstance(out, tuple):
            # for actor-critic and policy gradient approaches
            logits = out[0]
            action = np.argmax(logits.cpu().numpy().flatten())
            ret = {'logits': out[0], 'sampled_action': out[1], 'log_prob': out[2], 
                'entropy': out[3], 'value': out[4], 'deterministic_action': action}
            return action, ret
        else:
            # for dqn approaches
            q = out
            q = out.detach().cpu().numpy().ravel()
            return np.argmax(q), {'logits': q}

    def deterministic_episode(self):
        epi_info = {'logits': [], 'sampled_action': [], 'log_prob': [], 'entropy': [],
            'value': [], 'deterministic_action': [], 'reward': [], 'terminal': []}


        #env = self.config.evaluation_env
        env = self.evaluation_env
        state = env.reset()
        task_label = env.get_task()['task_label']

        # generate mask
        self._generate_nm_mask(task_label)

        total_rewards = 0
        while True:
            action, output_info = self.evaluation_action(state, task_label)
            state, reward, done, _ = env.step(action)
            total_rewards += reward
            for k, v in output_info.items(): epi_info[k].append(v)
            epi_info['reward'].append(reward)
            epi_info['terminal'].append(done)
            if done: break
        return total_rewards, epi_info

# rather than have multiple nm nets (one for each layer of the target network), we use one nm net
# that generate output to match all parameters of the target network.
class PPOAgentSCPModulatedFP_v2(PPOContinualLearnerAgent):
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

        # neuromodulated (hypernet) mask for modulated forward pass (FP)
        tasks = self.config.cl_tasks_info
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])
        self.nm_mask = {}
        self.mask_real = {}
        total_params = 0
        for n, p in self.network.named_parameters():
            self.nm_mask[n] = torch.zeros_like(p).to(self.config.DEVICE)
            self.mask_real[n] = torch.zeros_like(p).to(self.config.DEVICE)
            total_params += p.numel()
        self.nm_nets = NMNetBig(total_params, label_dim)
        #self.nm_nets = NMNetKWinnersBig(total_params, label_dim)
        self.nm_nets.to(config.DEVICE)

        # pm of nm nets
        self.nm_pm = {}
        self.nm_means = {}
        for n, p in self.nm_nets.named_parameters():
            p = deepcopy(p)
            p.data.zero_()
            self.nm_pm[n] = p.data.to(config.DEVICE)
        for n, p in self.nm_nets.named_parameters():
            p = deepcopy(p)
            p.data.zero_()
            self.nm_means[n] = p.data.to(config.DEVICE)

        all_parameters = []
        all_parameters += list(self.network.parameters())
        all_parameters += list(self.nm_nets.parameters())
        self.opt = config.optimizer_fn(all_parameters, config.lr)

    def iteration(self):
        #task_label = self.task.get_task()['task_label'] # current task
        task_label = self.task_label_param
        # NOTE generate mask
        self._generate_nm_mask(task_label)

        config = self.config
        rollout = []
        states = self.states
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label, masks=self.nm_mask)
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
            if config.cl_preservation != 'ewc': self.data_buffer.feed_batch([states, ])

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label, masks=self.nm_mask)[-2]
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
                #batch_task_label = np.repeat(task_label.reshape(1, -1), batch_dim, axis=0)
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, masks=self.nm_mask)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                weight_pres_loss = self.penalty()

                self.opt.zero_grad()

                # loss continuous mask
                (policy_loss + value_loss + weight_pres_loss).backward()

                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_.detach().cpu().numpy())
                self.opt.step()

                # NOTE generate mask for next batch of training
                self._generate_nm_mask(task_label)

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_).mean()

    def _generate_nm_mask(self, task_label):
        total_mask = self.nm_nets(task_label.reshape(1, -1))
        total_mask = total_mask.view(-1,)
        counter = 0
        for n, p in self.network.named_parameters():
            p_numel = p.numel()
            p_mask = total_mask[counter : counter + p_numel]
            mr = p_mask.view(p.shape)
            counter += p_numel

            if mr.requires_grad is True: 
                mr.retain_grad()
            self.mask_real[n] = mr
            # m1: binarize
            #mb = torch.sign(mr)
            # m2: continuous (sigmoided)
            #mb = torch.zeros_like(mr)
            #mb[mr > 0.] = torch.sigmoid(mr[mr > 0.] + 5.)
            # m3: continuous raw
            mb = mr

            if mb.requires_grad is True:
                mb.retain_grad()
            self.nm_mask[n] = mb

    def penalty(self):
        loss = 0
        for n, p in self.nm_nets.named_parameters():
            _loss = self.nm_pm[n] * (p - self.nm_means[n]) ** 2
            loss += _loss.sum()
        loss = loss * self.config.cl_loss_coeff
        return loss

    def consolidate(self, batch_size=32):
        print('sanity check, in scp consolidation with modulated fp')
        config = self.config
        
        #task_label = self.task.get_task()['task_label'] # current task
        #task_label = task_label.reshape(1, -1)
        task_label = self.task_label_param.detach()
        #task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_size, dim=0)
        task_label = task_label.reshape(1, -1)

        pm = {}
        for n, p in self.nm_nets.named_parameters():
            pm[n] = deepcopy(p)
            pm[n].data.zero_()
            pm[n] = pm[n].data.to(config.DEVICE)
        net.zero_grad()
        outs = net(task_label) # forward pass
        outs = outs.view(1, -1)
        outs_mean = outs.type(torch.float32).mean(dim=0)
        K = outs_mean.shape[0]
        for _ in range(config.cl_n_slices):
            xi = torch.randn(K, ).to(config.DEVICE)
            xi /= torch.sqrt((xi**2).sum())
            net.zero_grad()
            loss = torch.matmul(outs_mean, xi)
            loss.backward(retain_graph=True)
            # Update the temporary precision matrix
            for n, p in net.named_parameters():
                pm[n].data += p.grad.data ** 2

        for n, p in net.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.nm_pm[n] = config.cl_alpha*self.nm_pm[n] + (1 - config.cl_alpha) * pm[n]
            # Update the means
            self.nm_means[n] = deepcopy(p.data).to(config.DEVICE)

        # this return values are not useful for the class. only added for compatibility purpose.
        # consolidation is not applied to the target network.
        return self.precision_matrices, self.precision_matrices

    def evaluation_action(self, state, task_label):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        task_label = np.stack([task_label])
        out = self.network.predict(state, task_label=task_label, masks=self.nm_mask)
        self.config.state_normalizer.unset_read_only()
        if isinstance(out, dict) or isinstance(out, list) or isinstance(out, tuple):
            # for actor-critic and policy gradient approaches
            logits = out[0]
            action = np.argmax(logits.cpu().numpy().flatten())
            ret = {'logits': out[0], 'sampled_action': out[1], 'log_prob': out[2], 
                'entropy': out[3], 'value': out[4], 'deterministic_action': action}
            return action, ret
        else:
            # for dqn approaches
            q = out
            q = out.detach().cpu().numpy().ravel()
            return np.argmax(q), {'logits': q}

    def deterministic_episode(self):
        epi_info = {'logits': [], 'sampled_action': [], 'log_prob': [], 'entropy': [],
            'value': [], 'deterministic_action': [], 'reward': [], 'terminal': []}


        #env = self.config.evaluation_env
        env = self.evaluation_env
        state = env.reset()
        task_label = env.get_task()['task_label']

        # generate mask
        self._generate_nm_mask(task_label)

        total_rewards = 0
        while True:
            action, output_info = self.evaluation_action(state, task_label)
            state, reward, done, _ = env.step(action)
            total_rewards += reward
            for k, v in output_info.items(): epi_info[k].append(v)
            epi_info['reward'].append(reward)
            epi_info['terminal'].append(done)
            if done: break
        return total_rewards, epi_info
