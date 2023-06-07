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

#from ..network import *
#from ..component import *
#from .BaseAgent import *
#import torchvision


class TD3Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        self.task = None if config.task_fn is None else config.task_fn()
        '''if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
        tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
        tasks = [tasks_[task_id] for task_id in config.task_ids]
        del tasks_
        self.config.cl_tasks_info = self.task
        label_dim = None if not self.config.use_task_label else len(tasks[0]['task_label'])
        self.task_label_dim = label_dim'''


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


class TD3ContinualLearnerAgent(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)
        self.config = config
        self.task = None if config.task_fn is None else config.task_fn()
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
        tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
        tasks = [tasks_[task_id] for task_id in config.task_ids]
        del tasks_
        self.config.cl_tasks_info = self.task
        label_dim = None if not self.config.use_task_label else len(tasks[0]['task_label'])
        
        self.task_label_dim = label_dim

        random_seed(9157)

        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        _params = list(self.network.parameters())
        self.opt = config.optimizer_fn(_params, config.lr)
        self.total_steps = 0

        random_seed(config.seed)

        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn(self.task.action_dim)
        self.total_steps = 0

        self.running_episodes_rewards = [[] for _ in range(config.num_workers)]
        self.iteration_rewards = np.zeros(config.num_workers)

        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.layers_output = None
        self.data_buffer = Replay(memory_size=int(1e4), batch_size=256)

        self.curr_train_task_label = None
        self.curr_eval_task_label = None

        # other performance metric (specifically for metaworld environment)
        if self.task.name == config.ENV_METAWORLD or self.task.name == config.ENV_CONTINUALWORLD:
            self._rollout_fn = self._rollout_metaworld
            self.episode_success_rate = np.zeros(config.num_workers)
            self.last_episode_success_rate = np.zeros(config.num_workers)
            # used to compute average across all episodes that may occur in an iteration
            self.running_episodes_success_rate = [[] for _ in range(config.num_workers)]
            self.iteration_success_rate = np.zeros(config.num_workers)
        else:
            self._rollout_fn = self._rollout_normal
            self.episode_success_rate = None
            self.last_episode_success_rate = None
            self.running_episodes_success_rate = None
            self.iteration_success_rate = None

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



class TD3LLAgent(TD3ContinualLearnerAgent):
    '''
    PPO continual learning agent using supermask superposition algorithm
    task oracle available: agent informed about task boundaries (i.e., when
    one task ends and the other begins)

    supermask lifelong learning algorithm: https://arxiv.org/abs/2006.14769
    '''
    def __init__(self, config):
        TD3ContinualLearnerAgent.__init__(self, config)
        self.seen_tasks = {} # contains task labels that agent has experienced so far.
        self.new_task = False
        self.curr_train_task_label = None
        self.curr_eval_task_label = None

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
            # new task. add it to the agent's seen_tasks dictionary
            task_idx = len(self.seen_tasks) # generate an internal task index for new task
            self.seen_tasks[task_idx] = task_label
            self.new_task = True
            set_model_task(self.network, task_idx, new_task=True)
        else:
            set_model_task(self.network, task_idx)
        self.curr_train_task_label = task_label
        return

    def task_train_end(self):
        # NOTE, comment/uncomment alongside a block of code in `_forward_mask_lnear_comb` method in
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
            # internally for the task not yet seen.
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
            task_idx = self._label_to_idx(self.curr_train_task_label)
            set_model_task(self.network, task_idx)
        return
    

class TD3ShELLDP(TD3LLAgent):
    '''
    Lifelong learning (ppo continual learning with supermask) agent in ShELL
    settings. All agents executing in a distributed (multi-) process (DP) setting.
    '''
    def __init__(self, config):
        TD3LLAgent.__init__(self, config)
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
        # TODO fix algorithm. Current solution invloves selecting the first mask 
        # that is not Noneand using it as the knowledge from which the agent can 
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

        #task_label = self.curr_train_task_label
        task_idx = self._label_to_idx(task_label)

        # Process the single mask as opposed to multiple
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