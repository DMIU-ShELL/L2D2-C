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
        tasks = self.task.get_all_tasks(config.cl_requires_task_label)
        config.cl_num_tasks = len(tasks)
        self.config.cl_tasks_info = tasks
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])

        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        _params = list(self.network.parameters())
        self.opt = config.optimizer_fn(_params, config.lr)
        self.total_steps = 0
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.layers_output = None
        self.data_buffer = Replay(memory_size=int(1e4), batch_size=256)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        task_label = self.task.get_task()['task_label']
        task_label = tensor(task_label)
        batch_dim = config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        # TODO: review whether a data buffer is needed for the SS algorithm
        self.data_buffer.feed_batch([states, ])
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
            # TODO: review whether a data buffer is needed for the SS algorithm
            self.data_buffer.feed_batch([states, ])

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

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norms_.append(norm_.detach().cpu().numpy())
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return np.mean(grad_norms_)

    def task_train_start(self, task_idx):
        return

    def task_train_end(self, task_idx):
        return

class PPOAgentBaseline(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent baseline (with forgetting)
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

class PPOAgentSS(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using supermask superposition algorithm
    https://arxiv.org/abs/2006.14769
    '''
    def __init__(self, config):
       PPOContinualLearnerAgent.__init__(self, config)

    def task_train_start(self, task_idx):
        set_model_task(self.network, task_idx)
        return

    def task_train_end(self, task_idx):
        cache_masks(self.network)
        set_num_tasks_learned(self.network, task_idx + 1)
        return

    def task_eval_start(self, task_idx):
        self.network.eval()
        set_model_task(self.network, task_idx)
        return

    def task_eval_end(self, task_idx):
        self.network.train()
        return
