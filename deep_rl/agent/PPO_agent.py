#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from ..shell_modules import *
from copy import deepcopy
import numpy as np

'''
Original PPO agent implementations for single agent, single process shell and distributed shell.
'''
class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        #self.config = config
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
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])
        self.task_label_dim = label_dim 

        # set seed before creating network to ensure network parameters are
        # same across all shell agents
        torch.manual_seed(config.seed)

        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        _params = list(self.network.parameters())
        self.opt = config.optimizer_fn(_params, config.lr)
        self.total_steps = 0

        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        # running reward: used to compute average across all episodes
        # that may occur in an iteration
        self.running_episodes_rewards = [[] for _ in range(config.num_workers)]
        self.iteration_rewards = np.zeros(config.num_workers)


        self.mask_to_rewards = dict()

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

    #def __getstate__(self):
    #    raise Exception

    def iteration(self):
        #select_device(0) # -1 is CPU, a positive integer is the index of GPU
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

        ####################### DATA COLLECTION (FORWARD PASS) ENDS HERE #######################




        ####################### OPTIMISATION PHASE STARTS HERE #######################


        ## PREPROCESSING FOR MODEL UPDATES
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


        ## OPTIMISATIONS

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
        # clear running performance buffers
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]

        config = self.config
        rollout = []
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.running_episodes_rewards[i].append(self.episode_rewards[i])
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)

            # save data to buffer for the detect module
            self.data_buffer.feed_batch([states, actions, rewards, terminals, next_states])

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
            self.data_buffer.feed_batch([states, actions, rewards, terminals, next_states])

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states

        # compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])
            self.iteration_success_rate[i] = self._avg_episodic_perf(self.running_episodes_success_rate[i])

        return states, rollout

    def _avg_episodic_perf(self, running_perf):
        if len(running_perf) == 0: return 0.
        else: return np.mean(running_perf)

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
        self.seen_tasks = {} # contains task labels that agent has experienced so far.
        self.new_task = False
        self.curr_train_task_label = None

    def _label_to_idx(self, task_label):
        eps = 1e-5
        found_task_idx = None
        for task_idx, seen_task_label in self.seen_tasks.items():
            if np.linalg.norm((task_label - seen_task_label), ord=2) < eps:
                found_task_idx = task_idx
                break
        return found_task_idx
        
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


'''
Multiprocessing implementations of the above PPO agents. Data Collection and Optimisation can take place in parallel.
Uses pipes to communicate data between the main process and the agent's functions.
'''
import multiprocess as mp
import dill
dill.settings["recurse"] = True

# PPO_Agent
class PPOContinualLearnerAgent_mp(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)
        self.config = config
        self.task = None if config.task_fn is None else config.task_fn()
        #self.task = ParallelizedTask(MiniGridFlatObs(name, env_config_path, config.log_dir, config.seed, False), config.num_workers, log_dir=config.log_dir, single_process=False)
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task
        tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
        tasks = [tasks_[task_id] for task_id in config.task_ids]
        del tasks_
        self.config.cl_tasks_info = tasks
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])
        self.task_label_dim = label_dim 

        # set seed before creating network to ensure network parameters are
        # same across all shell agents
        torch.manual_seed(config.seed)
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        _params = list(self.network.parameters())
        self.opt = config.optimizer_fn(_params, config.lr)
        self.total_steps = 0

        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        # running reward: used to compute average across all episodes
        # that may occur in an iteration
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
    '''
    Entry point into the data collection loop. Get the configuration and states and run
    the rollout function defined in initalisation. The rollout function changes dependant on
    which environment is used (i.e. MetaWorld/ContinualWorld or MiniGrid/CTGraph)
    '''
    def data_collection(self, data_conn_parent):
        print(torch.cuda.is_initialized())
        select_device(0) # -1 is CPU, a positive integer is the index of GPU
        print(torch.cuda.is_initialized())
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

        # Carry out the rollout function as specified in the initialisation
        #states, rollout = self._rollout_fn(states, batch_task_label, data_conn_parent)
        self._rollout_fn(states, batch_task_label, data_conn_parent)

        # Send all data into the pipeline entrypoint
        #data_conn_parent.send(states, rollout, batch_task_label)

    '''
    Handles the optimisation phase. Recieves states, rollout, batch_task_label from the
    connection pipeline and
    '''
    def optimisation(self, data_conn_child, dict_conn_parent):
        select_device(0) # -1 is CPU, a positive integer is the index of GPU
        # Get all necessary data from the data collection pipe
        states, rollout, batch_task_label = data_conn_child.recv()


        ####################### OPTIMISATION PHASE STARTS HERE #######################

        ## PREPROCESSING FOR MODEL UPDATES
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


        ## OPTIMISATIONS

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


        # Send dictionary for logging output back to main process.
        dict_conn_parent.send([{'grad_norm': grad_norm_log, 'policy_loss': policy_loss_log, \
            'value_loss': value_loss_log, 'log_prob': log_probs_log, 'entropy': entropy_log, \
            'ppo_ratio': ratio_log}])
    
    def _rollout_normal(self, states, batch_task_label, data_conn_parent):
        # clear running performance buffers
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]

        config = self.config
        rollout = []
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.running_episodes_rewards[i].append(self.episode_rewards[i])
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)

            # save data to buffer for the detect module
            self.data_buffer.feed_batch([states, actions, rewards, terminals, next_states])

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states

        # compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])

        
        data_conn_parent.send([states, rollout, batch_task_label])
        #return states, rollout

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
            self.data_buffer.feed_batch([states, actions, rewards, terminals, next_states])

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states

        # compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])
            self.iteration_success_rate[i] = self._avg_episodic_perf(self.running_episodes_success_rate[i])

        return states, rollout

    def _avg_episodic_perf(self, running_perf):
        if len(running_perf) == 0: return 0.
        else: return np.mean(running_perf)

class LLAgent_mp(PPOContinualLearnerAgent_mp):
    '''
    PPO continual learning agent using supermask superposition algorithm
    task oracle available: agent informed about task boundaries (i.e., when
    one task ends and the other begins)

    supermask lifelong learning algorithm: https://arxiv.org/abs/2006.14769
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent_mp.__init__(self, config)
        self.seen_tasks = {} # contains task labels that agent has experienced so far.
        self.new_task = False
        self.curr_train_task_label = None

    def _label_to_idx(self, task_label):
        eps = 1e-5
        found_task_idx = None
        for task_idx, seen_task_label in self.seen_tasks.items():
            if np.linalg.norm((task_label - seen_task_label), ord=2) < eps:
                found_task_idx = task_idx
                break
        return found_task_idx
        
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
    #class CustomProcess(mp.Process):
    #    def __init__(self, task, seed)

    '''
    Lifelong learning (ppo continual learning with supermask) agent in ShELL
    settings. All agents executing in a distributed (multi-) process (DP) setting.
    '''
    def __init__(self, config):
        LLAgent_mp.__init__(self, config)
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


'''
Multiprocessing implementation of PPO agent using a custom process wrapper. Data collection and optimisation take place
together through iteration(), like the original code. Using the constructor of, generate the Model (network function) and
select the CUDA device for just this agent. Using a process wrapper, run the iteration() function.

In the future this approach needs to be revisited to consider using semaphores and events.
Further ideas:
 -> Using a Pool to run agent functions.
'''
import shutil
import time

class PPOContinualLearnerAgent_mpu(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)

        # Initialise CUDA inside the iteration
        select_device(0) # -1 is CPU, a positive integer is the index of GPU


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
        label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label'])
        self.task_label_dim = label_dim 

        # set seed before creating network to ensure network parameters are
        # same across all shell agents
        torch.manual_seed(config.seed)


        #### CREATE NETWORK FUNCTION FROM INSIDE PPOAGENT INSTEAD OF FROM MAIN PROCESS
        #self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        self.network = CategoricalActorCriticNet_SS(
            self.state_dim, self.action_dim, label_dim,
            phi_body=FCBody_SS(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200), num_tasks=num_tasks),
            actor_body=DummyBody_CL(200),
            critic_body=DummyBody_CL(200),
            num_tasks=num_tasks)


        _params = list(self.network.parameters())
        self.opt = config.optimizer_fn(_params, config.lr)
        self.total_steps = 0

        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        # running reward: used to compute average across all episodes
        # that may occur in an iteration
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

    def iteration(self):
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
        ####################### DATA COLLECTION (FORWARD PASS) ENDS HERE #######################



        ####################### OPTIMISATION PHASE STARTS HERE #######################
        ## PREPROCESSING FOR MODEL UPDATES
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


        ## OPTIMISATIONS

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
        # clear running performance buffers
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]

        config = self.config
        rollout = []
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.running_episodes_rewards[i].append(self.episode_rewards[i])
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)

            # save data to buffer for the detect module
            self.data_buffer.feed_batch([states, actions, rewards, terminals, next_states])

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
            self.data_buffer.feed_batch([states, actions, rewards, terminals, next_states])

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states

        # compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])
            self.iteration_success_rate[i] = self._avg_episodic_perf(self.running_episodes_success_rate[i])

        return states, rollout

    def _avg_episodic_perf(self, running_perf):
        if len(running_perf) == 0: return 0.
        else: return np.mean(running_perf)

class LLAgent_mpu(PPOContinualLearnerAgent_mpu):
    '''
    PPO continual learning agent using supermask superposition algorithm
    task oracle available: agent informed about task boundaries (i.e., when
    one task ends and the other begins)

    supermask lifelong learning algorithm: https://arxiv.org/abs/2006.14769
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent_mp.__init__(self, config)
        self.seen_tasks = {} # contains task labels that agent has experienced so far.
        self.new_task = False
        self.curr_train_task_label = None

    def _label_to_idx(self, task_label):
        eps = 1e-5
        found_task_idx = None
        for task_idx, seen_task_label in self.seen_tasks.items():
            if np.linalg.norm((task_label - seen_task_label), ord=2) < eps:
                found_task_idx = task_idx
                break
        return found_task_idx
        
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

class AgentProcess(mp.Process):
    ITERATION = 0
    GET_TASKS = 1
    GET_TASK_NAME = 2
    RUN_RESET_TASK = 3
    DATA_BUFFER_CLEAR = 4
    TASK_TRAIN_START = 5
    GET_TOTAL_STEPS = 6
    GET_ITERATION_REWARDS = 7
    GET_LAST_EPISODE_REWARDS = 8
    SET_STATES = 9
    GET_AGENT_NAME = 10
    SAVE = 11
    NAMED_PARAMS = 12
    GET_ATTR = 13
    GET_LAYERS_OUTPUT = 14
    TASK_TRAIN_END = 15
    TASK_EVAL_END = 16
    TASK_EVAL_START = 17
    SET_EVAL_STATES = 18
    RESET_EVAL_ENV = 19
    EVALUATE_CL = 20
    CLOSE = 21

    def __init__(self, pipe_child, pipe_parent, Agent, config):
        mp.Process.__init__(self)
        # Get the pipe child
        self.pipe_child = pipe_child
        self.pipe_parent = pipe_parent
        # Make instance of an Agent object i.e., LLAgent_mpu
        self.agent = Agent(config)

        config.agent_name = self.agent.__class__.__name__
        self.tasks = self.agent.config.cl_tasks_info
        config.cl_num_learn_blocks = 1

        #shutil.copy(config.env_config_path, config.log_dir + '/env_config.json')
        #with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        #    pickle.dump(self.tasks, f)

    def run(self):
        while True:
            op, data = self.pipe_child.recv()
            if op == self.ITERATION:
                # Run the LLAgent iteration() function and return loggin dictionary
                # through the pipe child.
                result = self.agent.iteration()
                self.pipe_parent.send(result)

            elif op == self.GET_TASKS:
                self.pipe_parent.send(self.tasks)

            elif op == self.GET_TASK_NAME:
                self.pipe_parent.send(self.agent.task.name)

            elif op == self.RUN_RESET_TASK:
                self.pipe_parent.send(self.agent.task.reset_task(data))

            elif op == self.DATA_BUFFER_CLEAR:
                self.pipe_parent.send(self.agent.data_buffer.clear())

            elif op == self.TASK_TRAIN_START:
                self.agent.task_train_start(data)

            elif op == self.GET_TOTAL_STEPS:
                self.pipe_parent.send(self.agent.total_steps)

            elif op == self.GET_ITERATION_REWARDS:
                self.pipe_parent.send(self.agent.iteration_rewards)

            elif op == self.GET_LAST_EPISODE_REWARDS:
                self.pipe_parent.send(self.agent.last_episode_rewards)

            elif op == self.SET_STATES:
                self.agent.states = data

            elif op == self.GET_AGENT_NAME:
                self.pipe_parent.send(self.agent.__class__.__name__)

            elif op == self.SAVE:
                self.agent.save(data)

            elif op == self.NAMED_PARAMS:
                self.pipe_parent.send(list(self.agent.network.named_parameters()))

            elif op == self.GET_ATTR:
                self.pipe_parent.send(dir(self.agent))

            elif op == self.GET_LAYERS_OUTPUT:
                self.pipe_parent.send(self.agent.layers_output)

            elif op == self.TASK_TRAIN_END:
                self.pipe_parent.send(self.agent.task_train_end())

            elif op == self.TASK_EVAL_START:
                self.agent.task_eval_start(data)

            elif op == self.TASK_EVAL_END:
                self.agent.task_eval_end()

            elif op == self.SET_EVAL_STATES:
                self.agent.evaluation_states = data

            elif op == self.RESET_EVAL_ENV:
                self.pipe_parent.send(self.agent.evaluation_env.reset_task(data))

            elif op == self.EVALUATE_CL:
                perf, episodes = self.agent.evaluate_cl(data)
                self.pipe_parent.send([perf, episodes])

            elif op == self.CLOSE:
                self.agent.close()

            else:
                raise Exception('Unknown command')

class ParallelizedAgent:
    def __init__(self, Agent, config):
        self.pipe1, worker_pipe1 = mp.Pipe([False])
        pipe2, self.worker_pipe2 = mp.Pipe([False])
        self.worker = AgentProcess(worker_pipe1, pipe2, Agent, config)
        self.worker.start()

    def iteration(self):
        # Send OPCODE to AgentWrapper to run LLAgent_mpu.iteration()
        self.pipe1.send([AgentProcess.ITERATION, None])

    def get_tasks(self):
        self.pipe1.send([AgentProcess.GET_TASKS, None])
        return self.worker_pipe2.recv()

    def get_task_name(self):
        self.pipe1.send([AgentProcess.GET_TASK_NAME, None])
        return self.worker_pipe2.recv()

    def task_reset_task(self, data):
        self.pipe1.send([AgentProcess.RUN_RESET_TASK, data])
        return self.worker_pipe2.recv()

    def data_buffer_clear(self):
        self.pipe1.send([AgentProcess.DATA_BUFFER_CLEAR, None])

    def task_train_start(self, data):
        self.pipe1.send([AgentProcess.TASK_TRAIN_START, data])

    def get_total_steps(self):
        self.pipe1.send([AgentProcess.GET_TOTAL_STEPS, None])
        return self.worker_pipe2.recv()
    
    def get_iteration_rewards(self):
        self.pipe1.send([AgentProcess.GET_ITERATION_REWARDS, None])
        return self.worker_pipe2.recv()

    def get_last_episode_rewards(self):
        self.pipe1.send([AgentProcess.GET_LAST_EPISODE_REWARDS, None])
        return self.worker_pipe2.recv()

    def set_states(self, data):
        self.pipe1.send([AgentProcess.SET_STATES, data])

    def get_pipe(self):
        return self.worker_pipe2.recv()

    def get_agent_name(self):
        self.pipe1.send([AgentProcess.GET_AGENT_NAME, None])
        return self.worker_pipe2.recv()

    def save(self, data):
        self.pipe1.send([AgentProcess.SAVE, data])

    def get_named_parameters(self):
        self.pipe1.send([AgentProcess.NAMED_PARAMS, None])
        return self.worker_pipe2.recv()

    def get_attr(self):
        self.pipe1.send([AgentProcess.GET_ATTR, None])
        return self.worker_pipe2.recv()

    def get_layers_output(self):
        self.pipe1.send([AgentProcess.GET_LAYERS_OUTPUT, None])
        return self.worker_pipe2.recv()

    def task_train_end(self):
        self.pipe1.send([AgentProcess.TASK_TRAIN_END, None])
        return self.worker_pipe2.recv()

    def task_eval_start(self, data):
        self.pipe1.send([AgentProcess.TASK_EVAL_START, data])

    def task_eval_end(self):
        self.pipe1.send([AgentProcess.TASK_EVAL_END, None])

    def set_evaluation_states(self, data):
        self.pipe1.send([AgentProcess.SET_EVAL_STATES, data])

    def evaluation_env_reset_task(self, data):
        self.pipe1.send([AgentProcess.RESET_EVAL_ENV, data])

    def evaluate_cl(self, data):
        self.pipe1.send([AgentProcess.EVALUATE_CL, data])

    def close(self):
        self.pipe1.send([AgentProcess.CLOSE, None])

    def kill(self):
        self.worker.join()
        self.worker.close()