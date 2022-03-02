#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
from ..utils import *

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.evaluation_env = self.config.evaluation_env
        if self.evaluation_env is not None:
            self.evaluation_state = self.evaluation_env.reset()
            self.evaluation_return = 0

    def close(self):
        if hasattr(self.task, 'close'):
            self.task.close()
        if hasattr(self.evaluation_env, 'close'):
            self.evaluation_env.close()

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        action = self.network.predict(state, to_numpy=True)
        self.config.state_normalizer.unset_read_only()
        return np.argmax(action.flatten())

    def deterministic_episode(self):
        env = self.config.evaluation_env
        state = env.reset()
        total_rewards = 0
        while True:
            action = self.evaluation_action(state)
            state, reward, done, _ = env.step(action)
            total_rewards += reward
            if done:
                break
        return total_rewards

    def evaluation_episodes(self):
        interval = self.config.evaluation_episodes_interval
        if not interval or self.total_steps % interval:
            return
        rewards = []
        for ep in range(self.config.evaluation_episodes):
            rewards.append(self.deterministic_episode())
        self.config.logger.info('evaluation episode return: %f(%f)' % (
            np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))))

    def evaluate(self, steps=1):
        config = self.config
        if config.evaluation_env is None or self.config.evaluation_episodes_interval:
            return
        for _ in range(steps):
            action = self.evaluation_action(self.evaluation_state)
            self.evaluation_state, reward, done, _ = self.evaluation_env.step(action)
            self.evaluation_return += reward
            if done:
                self.evaluation_state = self.evaluation_env.reset()
                self.config.logger.info('evaluation episode return: %f' % (self.evaluation_return))
                self.evaluation_return = 0

class BaseContinualLearnerAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)

    def consolidate(self, config=None):
        raise NotImplementedError

    def penalty(self):
        raise NotImplementedError

    def evaluation_action(self, state, task_label):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        task_label = np.stack([task_label])
        out = self.network.predict(state, task_label=task_label)
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
        #task_label = env.get_task()['task_label']
        task_label = env.get_task()['task_label_agent']
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

    def evaluate_cl(self, num_iterations=100):
        # evaluation method for continual learning agents
        rewards = []
        episodes = []
        with torch.no_grad():
            for ep in range(num_iterations):
                total_episode_reward, episode_info = self.deterministic_episode()
                rewards.append(total_episode_reward)
                episodes.append(episode_info)
        return rewards, episodes
