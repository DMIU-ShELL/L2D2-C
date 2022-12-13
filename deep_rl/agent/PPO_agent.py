#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


#   ___________.__                 _____                            __   
#   \__    ___/|  |__    ____     /  _  \    ____    ____    ____ _/  |_ 
#     |    |   |  |  \ _/ __ \   /  /_\  \  / ___\ _/ __ \  /    \\   __\
#     |    |   |   Y  \\  ___/  /    |    \/ /_/  >\  ___/ |   |  \|  |  
#     |____|   |___|  / \___  > \____|__  /\___  /  \___  >|___|  /|__|  
#                   \/      \/          \//_____/       \/      \/       
#

from copy import deepcopy
import multiprocess as mp
import torch
import torch.nn as nn
import numpy as np
from queue import Empty
#from ..network import *
#from ..network.network_heads import CategoricalActorCriticNet_SS
from ..network import network_heads as nethead
#from ..network import *
#from ..component import *
#from .BaseAgent import *
from ..shell_modules.mmn.ssmask_utils import set_model_task, consolidate_mask, cache_masks, set_num_tasks_learned, get_mask, set_mask

from .BaseAgent import BaseAgent, BaseContinualLearnerAgent
from ..network.network_bodies import FCBody_SS, DummyBody_CL
from ..utils.torch_utils import random_seed, select_device, tensor
from ..utils.misc import Batcher
from ..component.replay import Replay
#from ..detect import Detect


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
        tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
        tasks = [tasks_[task_id] for task_id in config.task_ids]
        del tasks_
        self.config.cl_tasks_info = tasks
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label']) #CHANGE THAT FOR THE
        self.task_label_dim = label_dim

        '''#Create a Refernce for the Wasserstain Embeddings
        torch.manual_seed(98)
        reference = torch.rand(500, self.task.state_dim)'''

        #Variable for storing the detect reference number.
        self.detect_reference_num = config.detect_reference_num

        #Variable for storing the number of samples hyperperameter of the detect module.
        self.detect_num_samples = config.detect_num_samples

        #Varible for checking for dishiding wether the agent has encountered a new task or not.
        self.emb_dist_threshold = config.emb_dist_threshold

        #Variable for storing the frequency of the detect module activation.
        self.detect_module_activation_frequency = config.detect_module_activation_frequency

        #Create a list for saving the calculated embeddings
        self.encounterd_task_embs = []

        #Varible for storing the current embedding label as an attribute of the agent itself.
        self.current_task_emb =  None #self.task.get_task()['task_emb']

        self.new_task_emb = None

        # set seed before creating network to ensure network parameters are
        # same across all shell agents
        #torch.manual_seed(config.seed)
        
        random_seed(9157)   # Chris

        #Precalculate the embedding size based on the reference and the network observation size.
        tmp_state_obs  = self.task.reset()
        tmp_state_obs = config.state_normalizer(tmp_state_obs)
        observation_size = tmp_state_obs.shape[1]

        #Assing a detect Component to the Agent upon initialisation
        self.detect = config.detect_fn(self.detect_reference_num, observation_size, self.detect_num_samples)


        #Variable for saving the size of the task embedding that the detect module has produced.
        #Initially we store the precalculated embedding size
        self.detect.set_reference(observation_size, self.detect_reference_num, self.task.action_dim)

        self.task_emb_size  = self.detect.precalculate_embedding_size(self.detect_reference_num, observation_size, self.task.action_dim)
        self.current_task_emb = torch.zeros(self.get_task_emb_size())
        self.new_task_emb =  torch.zeros(self.get_task_emb_size())
        print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII:", observation_size, self.task_emb_size)
        #Ihave changed the bellow commande by substitue the label dim with embedding dim
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, self.task_emb_size)
        _params = list(self.network.parameters())
        self.opt = config.optimizer_fn(_params, config.lr)
        self.total_steps = 0

        random_seed(config.seed)    # Chris


        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        # running reward: used to compute average across all episodes
        # that may occur in an iteration
        self.running_episodes_rewards = [[] for _ in range(config.num_workers)]
        self.iteration_rewards = np.zeros(config.num_workers)

        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

        
        self.layers_output = None
        self.data_buffer = Replay(memory_size=int(1e4), batch_size=512)

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

    def iteration(self):
        '''This function performs the training iteration.
        It is where the learner of the agent (the neural network) is being optimized'''
        config = self.config
        rollout = []
        states = self.states
        if self.curr_train_task_label is not None:
            task_label = self.curr_train_task_label
        else:
            task_label = self.task.get_task()['task_label']
            assert False, 'manually set (temporary) breakpoint. code should not get here.'

        task_label = tensor(task_label)
        batch_dim = config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        #print(batch_task_label.shape)
        states, rollout = self._rollout_fn(states, batch_task_label)

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
        eps = 1e-6
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        grad_norm_log = []
        policy_loss_log = []
        value_loss_log = []
        log_probs_log = []
        entropy_log = []
        ratio_log = []
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

                log_probs_log.append(log_probs.detach().cpu().numpy().mean())
                entropy_log.append(entropy_loss.detach().cpu().numpy().mean())
                ratio_log.append(ratio.detach().cpu().numpy().mean())
                policy_loss_log.append(policy_loss.detach().cpu().numpy())
                value_loss_log.append(value_loss.detach().cpu().numpy())

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norm_log.append(norm_.detach().cpu().numpy())
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return {'grad_norm': grad_norm_log, 'policy_loss': policy_loss_log, \
            'value_loss': value_loss_log, 'log_prob': log_probs_log, 'entropy': entropy_log, \
            'ppo_ratio': ratio_log}


    def iteration_emb(self):
        '''This function performs the training iteration.
        It is where the learner of the agent (the neural network) is being optimized.
        This method is using task embeddings instead of task labels.'''
        config = self.config
        rollout = []
        states = self.states
        if self.curr_train_task_label is not None:
            task_label = self.curr_train_task_label
        else:
            task_label = self.task.get_task()['task_label']
            assert False, 'manually set (temporary) breakpoint. code should not get here.'

        task_label = tensor(task_label)
        batch_dim = config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        states, rollout = self._rollout_fn(states, batch_task_label)

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
        eps = 1e-6
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        grad_norm_log = []
        policy_loss_log = []
        value_loss_log = []
        log_probs_log = []
        entropy_log = []
        ratio_log = []
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

                log_probs_log.append(log_probs.detach().cpu().numpy().mean())
                entropy_log.append(entropy_loss.detach().cpu().numpy().mean())
                ratio_log.append(ratio.detach().cpu().numpy().mean())
                policy_loss_log.append(policy_loss.detach().cpu().numpy())
                value_loss_log.append(value_loss.detach().cpu().numpy())

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norm_log.append(norm_.detach().cpu().numpy())
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return {'grad_norm': grad_norm_log, 'policy_loss': policy_loss_log, \
            'value_loss': value_loss_log, 'log_prob': log_probs_log, 'entropy': entropy_log, \
            'ppo_ratio': ratio_log}

    def _rollout_normal(self, states, batch_task_label):
        '''It runs the agent on the environment and collects SAR data to staore in the Replay
        Buffer'''
        # clear running performance buffers
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]

        config = self.config
        rollout = []
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label) #NOTE HERE WE BREAK in iteration 315!!!!!!!!!!!!!!!!!!!!!!!
            
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.running_episodes_rewards[i].append(self.episode_rewards[i])
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            #print(actions.cpu())
            # save data to buffer for the detect module
            self.data_buffer.feed_batch([states, actions.cpu(), rewards, terminals, next_states])

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states

        # compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])

        return states, rollout

    # rollout for metaworld and continualworld environments. it is similar to normal
    # rollout with the inclusion of the capture of success rate metric.
    def _rollout_metaworld(self, states, batch_task_label):
        # clear running performance buffers
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]
        self.running_episodes_success_rate = [[] for _ in range(self.config.num_workers)]

        config = self.config
        rollout = []
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, infos = self.task.step(actions.cpu().detach().numpy())
            success_rates = [info['success'] for info in infos]
            self.episode_rewards += rewards
            self.episode_success_rate += success_rates
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.running_episodes_rewards[i].append(self.episode_rewards[i])
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
                    self.episode_success_rate[i] = (self.episode_success_rate[i] > 0).astype(np.uint8)
                    self.running_episodes_success_rate[i].append(self.episode_success_rate[i])
                    self.last_episode_success_rate[i] = self.episode_success_rate[i]
                    self.episode_success_rate[i] = 0
            next_states = config.state_normalizer(next_states)

            # save data to buffer for the detect module
            self.data_buffer.feed_batch([states, actions.cpu(), rewards, terminals, next_states])

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states

        # compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])
            self.iteration_success_rate[i] = self._avg_episodic_perf(self.running_episodes_success_rate[i])

        return states, rollout

    
    def sar_data_extraction(self):
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
        np.savetxt("FROM_SAR_WITH_LOVE.txt", 
           sar_data_arr,
           fmt='%f')
        print("SAR_DATA_NP_TYPE:", type(sar_data_arr))
        #print("SAR_DATA_NP_ARR:", sar_data_arr)
        return sar_data_arr

    def compute_task_embedding(self, some_sar_data):
        '''Function for computing the task embedding based on the current
        batch of SAR data derived from the replay buffer.'''
        #print("TASK NUM OF ACTIONS:", self.task.action_space.n)
        task_embedding = self.detect.lwe(some_sar_data)
        self.new_task_emb = task_embedding
        #self.current_task_emb = task_embedding
        #self.task.set_current_task_info('task_emb', task_embedding)
        #self.task.get_task()['task_emb'] = task_embedding
        self.task_emb_size = len(task_embedding)
        return task_embedding



    def get_task_emb_size(self):
        '''''A getter method for retreiving the task embedding size.'''
        return self.task_emb_size

    def set_task_emb_size(self, an_emb_size):
        '''A setter for dynamically setting the mebedding size.'''
        self.task_emb_size = an_emb_size


    def get_emb_dist_threshold(self):
        '''A getter method for reteiving the distnce threshold for the embeddings.'''
        return self.emb_dist_threshold

    def set_emb_dist_threshold(self, a_distance_threshold):
        '''A setter for dynamically setting the distance threshold of the embeddings.'''
        self.emb_dist_threshold = a_distance_threshold


    def get_detect_module_activation_frequency(self):
        '''A getteer for the detect module activation frequency'''
        return self.detect_module_activation_frequency

    def set_detect_module_activation_frequency(self, some_activation_frequncy):
        '''A setter method for manually setting the detect module activation frequnecy.'''
        self.detect_module_activation_frequncy = some_activation_frequncy


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

    def set_new_task_emb(self, a_new_emb):
        ''''''
        self.new_task_emb = a_new_emb


    def get_task_emb_size(self):
        '''A getter method for retreiving the task embedding size.'''
        return self.task_emb_size

    def set_task_emb_size(self, a_task_emb_size):
        '''A setter method for updating the task embedding size with a new
        embedding size (if the new task has different dimensionality in its observation.'''
        self.task_emb_size = a_task_emb_size



    def calculate_emb_distance(self, the_current_embedding, a_new_embedding):
        '''Method that calculates the distance of the newlly computed task embedding and
        compered to the existing one by calling the distance method of the agent's detect
        module.'''
        emb_dist = self.detect.emb_distance(the_current_embedding, a_new_embedding)
        
        return emb_dist


    def assign_task_emb(self, a_new_emb, emb_distance, an_emb_dist_threshold):
        '''It assigns the most up to date embedding to the current task based
        on the embedding distance thershold. 
            If the distance is smaller than the treshold it means that
        the agent is stil working at the same task. Hence we update the embeding
        by averaging the old ebedding, that the agent already has for that task,
        with the new embedding caluclated the detect module.o
            If the distance is bigger than the threshold then the agent encounters
        a  new task so we update the attribute 'emb' of the coresponding of the
        list of dictionaries that our agent poseses'''

        #Temp varible for checking if there is a key-value pair created in the info_dict of the current task
        #the agent tries to solve.
        key_to_check = 'task_emb'

        #Message String that indicates if the detecte module has spotted a task change or not.
        str_task_chng_msg = ''

        #Flag for task change detected by the detect module (True if task change happend, False otherwise).
        task_chng_flag = None

        #if the pair does not exist, the agent encounters this task for the first time, we create the pair
        #and we assing as an embedding value a torch.Tensor of zeros of the same size as the embedding the
        #detect module has calculatd.
        if not key_to_check in self.task.get_task().keys():
            self.task.set_current_task_info(key_to_check, torch.zeros(self.get_task_emb_size()))
            print("TASK INFO REGISTRY UPDATED WITH EMBEDDING KEY-VALUE PAIR!!!!!!!!!!!!!!!!")
        print("TASK INFO KEYS:", self.task.get_task(all_workers=True))
        if emb_distance < an_emb_dist_threshold:
             self.task.get_task()['task_emb'] = (self.task.get_task()['task_emb'] + a_new_emb) / 2
             self.set_current_task_embedding(self.task.get_task()['task_emb'])#saving the updated embedding value to the agent cur_emb attribute as well
             #return the current task idx using the hidden _embedding_to_idx method 
             current_task_idx = self._embedding_to_idx(self.get_current_task_embedding(), self.get_emb_dist_threshold())
             #update the embedding value on the dictionary of registered seen taks for the current task basked on the current task idx
             self.seen_tasks[current_task_idx] = self.get_current_task_embedding()
             str_task_chng_msg = "TASK CHNAGE NNNNOOOOTTTTT DETECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
             task_chng_flag = False
        else:
            self.task_train_end_emb()
            self.task_train_start_emb(a_new_emb, an_emb_dist_threshold)
            self.task.get_task()['task_emb'] = a_new_emb
            self.set_current_task_embedding(a_new_emb)#self.task.get_task()['task_emb'])#saving the updated embedding value to the agent cur_emb attribute as well
            self.set_new_task_emb(a_new_emb)
            str_task_chng_msg = "TASK CHNAGE DETECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            
            task_chng_flag = True

        return str_task_chng_msg, task_chng_flag


    def _avg_episodic_perf(self, running_perf):
        if len(running_perf) == 0: return 0.
        else: return np.mean(running_perf)

    
    def store_embeddings(self, an_embedding):
        '''Function that appends on the list of different embeddings the ebedding for 
        a new encountered task.'''
        self.encounterd_task_embs. append(an_embedding)

class BaselineAgent(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent baseline (experience catastrophic forgetting)
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

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

class LLAgent(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using supermask superposition algorithm
    task oracle available: agent informed about task boundaries (i.e., when
    one task ends and the other begins)

    supermask lifelong learning algorithm: https://arxiv.org/abs/2006.14769
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)
        self.seen_tasks = {} # contains task labels / embeddings that agent has experienced so far.
        self.new_task = False
        self.curr_train_task_label = None
        self.curr_eval_task_label = None
        #Varibles for tracking the current task embedding, influenced by the two varibles above, and working at a same fashion.
        self.curr_train_task_emb = None
        self.curr_eval_task_emb = None

    def _label_to_idx(self, task_label):
        eps = 1e-5
        found_task_idx = None
        for task_idx, seen_task_label in self.seen_tasks.items():
            if np.linalg.norm((task_label - seen_task_label), ord=2) < eps:#Chnage this with the embedding distance threshold Perhaps.
                found_task_idx = task_idx
                break
        return found_task_idx

    def _embedding_to_idx(self, a_task_embedding, an_embedding_embedding_distance_threshold):
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
        for task_idx, seen_task_embedding in self.seen_tasks.items():
            if np.linalg.norm((a_task_embedding - seen_task_embedding), ord=2) < an_embedding_embedding_distance_threshold:
                found_task_idx = task_idx
                break
        return found_task_idx

        
    def task_train_start(self, task_label):

        '''Method for starting the new <<Detected>> task. It takes as an argument the tasklabel given by the 
        the "trainer_shell" from the "shell_tasks" list of dictionaries.'''

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


    def task_train_start_emb(self,a_task_embedding, an_embedding_embedding_distance_threshold):
        '''Method for starting the training procedure upon a new Detected task form the Detect Module.
        It is based on the "task_train_start()" method.'''

        task_idx = self._embedding_to_idx(a_task_embedding, an_embedding_embedding_distance_threshold)
        if task_idx is None:
            # new task. add it to the agent's seen_tasks dictionary
            task_idx = len(self.seen_tasks) # generate an internal task index for new task
            self.seen_tasks[task_idx] = a_task_embedding
            self.new_task = True
            set_model_task(self.network, task_idx, new_task=True)
        else:
            set_model_task(self.network, task_idx)
            self.current_task_emb = a_task_embedding
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

class ShellAgent_SP(LLAgent):
    '''
    Lifelong learning (ppo continual learning with supermask) agent in ShELL
    settings. All agents executing in a single/uni process (SP) setting.
    '''
    def __init__(self, config):
        LLAgent.__init__(self, config)

    def _select_mask(self, agents, masks, ensemble=False):
        found_mask = None
        if ensemble:
            raise NotImplementedError
        else:
            for agent, mask in zip(agents, masks):
                if mask is not None:
                    found_mask = mask
                    break
        return found_mask

    def ping_agents(self, agents):
        task_label = self.task.get_task()['task_label']
        task_idx = self._label_to_idx(task_label)
        masks = [agent.ping_response(task_label) for agent in agents]
        masks_count = sum([1 if m is not None else 0 for m in masks])
        mask = self._select_mask(agents, masks)
        if mask is not None:
            # function from deep_rl/shell_modules/mmn/ssmask_utils.py
            set_mask(self.network, mask, task_idx)
            #return True
            return masks_count
        else:
            #return False
            return masks_count

    def ping_response(self, task_label):
        task_idx = self._label_to_idx(task_label)
        # get task mask.
        if task_idx is None:
            mask = None
        else:
            # function from deep_rl/shell_modules/mmn/ssmask_utils.py
            mask = get_mask(self.network, task_idx)
        return mask

class ShellAgent_DP(LLAgent):
    '''
    Lifelong learning (ppo continual learning with supermask) agent in ShELL
    settings. All agents executing in a distributed (multi-) process (DP) setting.
    '''
    def __init__(self, config):
        LLAgent.__init__(self, config)
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

class LLAgent_NoOracle(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using supermask superposition algorithm
    with *no task oracle*: agent is not informed about task boundaries 
    (i.e., when one task ends and the other begins) and has to detect task
    change by itself.

    supermask lifelong learning algorithm: https://arxiv.org/abs/2006.14769
    '''
    def __init__(self, config):
       PPOContinualLearnerAgent.__init__(self, config)
       self.seen_tasks = {} # contains task labels that agent has experienced so far.
       self.new_task = False
       self.curr_train_task_label = None

    def _name_to_idx(self, name):
        found_task_idx = None
        for task_idx, value in self.seen_tasks.items():
            seen_task_label, task_name = value
            if name == task_name:
                found_task_idx = task_idx
                break
        return found_task_idx

    def _label_to_idx(self, task_label):
        eps = 1e-5
        found_task_idx = None
        for task_idx, value in self.seen_tasks.items():
            seen_task_label, task_name = value
            if np.linalg.norm((task_label - seen_task_label), ord=2) < eps:
                found_task_idx = task_idx
                break
        return found_task_idx
        
    # TODO block of code to remove
    #def _select_mask(self, agents, masks, ensemble=False):
    #    found_mask = None
    #    if ensemble:
    #        raise NotImplementedError
    #    else:
    #        for agent, mask in zip(agents, masks):
    #            if mask is not None:
    #                found_mask = mask
    #                break
    #    return found_mask

    def update_task_label(self, task_label):
        # TODO: consider other ways to update the label as detect module
        # alters it. Maybe moving average?
        task_idx = self._label_to_idx(self.curr_train_task_label)
        self.seen_tasks[task_idx][0] = task_label
        self.curr_train_task_label = task_label

    def set_first_task(self, task_label, task_name):
        # start first task
        task_idx = 0 # first task idx is 0
        self.seen_tasks[task_idx] = [task_label, task_name]
        self.new_task = True
        set_model_task(self.network, task_idx)
        self.curr_train_task_label = task_label
        return

    def task_change_detected(self, task_label, task_name):
        # end current task (if any)
        if self.curr_train_task_label is not None:
            cache_masks(self.network)
            if self.new_task:
                set_num_tasks_learned(self.network, len(self.seen_tasks))
            self.new_task = False # reset flag
            self.curr_train_task_label = None

        # start next task
        # use task label or task name to check if task already exist in model
        task_idx = self._label_to_idx(task_label)
        if task_idx is None:
            # new task. add it to the agent's seen_tasks dictionary
            task_idx = len(self.seen_tasks) # generate an internal task index for new task
            self.seen_tasks[task_idx] = [task_label, task_name]
            self.new_task = True
        set_model_task(self.network, task_idx)
        self.curr_train_task_label = task_label
        return

    def task_eval_start(self, task_name):
        self.network.eval()
        task_idx = self._name_to_idx(task_name)
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
        return

    def task_eval_end(self):
        self.network.train()
        # resume training the model on train task label if training
        # was on before running evaluations.
        if self.curr_train_task_label is not None:
            task_idx = self._label_to_idx(self.curr_train_task_label)
            set_model_task(self.network, task_idx)
        return