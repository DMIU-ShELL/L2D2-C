#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            actions, log_probs, entropy, values = self.network.predict(config.state_normalizer(states))
            next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([log_probs, values, actions, rewards, 1 - terminals, entropy])
            states = next_states

        self.states = states
        pending_value = self.network.predict(config.state_normalizer(states))[-1]
        rollout.append([None, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            log_prob, value, actions, rewards, terminals, entropy = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [log_prob, value, returns, advantages, entropy]

        log_prob, value, returns, advantages, entropy = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob * advantages
        value_loss = 0.5 * (returns - value).pow(2)
        entropy_loss = entropy.mean()

        self.policy_loss = np.mean(policy_loss.cpu().detach().numpy())
        self.entropy_loss = np.mean(entropy_loss.cpu().detach().numpy())
        self.value_loss = np.mean(value_loss.cpu().detach().numpy())

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).mean().backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps

class A2CContinualLearnerAgent(BaseAgent, BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

        # continual learning (cl) setup
        self.params = {n: p for n, p in self.network.named_parameters() if p.requires_grad}
        self.precision_matrices = {}
        self.means = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.precision_matrices[n] = p.data.to(config.device)
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.means[n] = p.data.to(config.device)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        ret_states = [config.state_normalizer(states), ]
        for _ in range(config.rollout_length):
            actions, log_probs, entropy, values = self.network.predict(config.state_normalizer(states))
            next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([log_probs, values, actions, rewards, 1 - terminals, entropy])
            states = next_states
            ret_states.append(config.state_normalizer(states))

        self.states = states
        pending_value = self.network.predict(config.state_normalizer(states))[-1]
        rollout.append([None, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            log_prob, value, actions, rewards, terminals, entropy = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [log_prob, value, returns, advantages, entropy]

        log_prob, value, returns, advantages, entropy = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob * advantages
        value_loss = 0.5 * (returns - value).pow(2)
        entropy_loss = entropy.mean()

        self.policy_loss = np.mean(policy_loss.cpu().detach().numpy())
        self.entropy_loss = np.mean(entropy_loss.cpu().detach().numpy())
        self.value_loss = np.mean(value_loss.cpu().detach().numpy())

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).mean().backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps

        return np.concatenate(ret_states, axis=0)

    def consolidate(self, data, batch_size=32):
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.device)

        # Set the model in the evaluation mode
        self.network.eval()

        # Get network outputs
        if isinstance(data, list): data = np.array(data)
        idxs = np.arange(len(data))
        np.random.shuffle(idxs)
        num_batches = len(data) // batch_size
        #z = []
        #for batch_idx in range(num_batches):
        #    start, end = batch_idx * batch_size, (batch_idx+1) * batch_size
        #    states = data[start:end ,...]
        #    z.append(self.network(states))            
        #z = torch.cat(z)
        #zmean = z.mean(dim=0)
        #K = zmean.shape[0]
        #xi = torch.stack([(xi_/torch.sqrt((xi_**2).sum())).to(self.device) \
        #       for xi_ in torch.randn((K,self.n_slices))])
        #for l in tqdm(range(config.cl_n_slices)):
        #    self.network.zero_grad()
        #    out = torch.matmul(zmean,xi[:,l])
        #    out.backward(retain_graph=True)

        #    # Update the temporary precision matrix
        #    for n, p in self.model.named_parameters():
        #        precision_matrices[n].data+=p.grad.data ** 2 / float(len(data) * config.cl_n_slices)
        for batch_idx in range(num_batches):
            start, end = batch_idx * batch_size, (batch_idx+1) * batch_size
            states = data[start:end, ...]
            actions, _, _, values = self.network.predict(states)
            actions_mean = actions.mean(dim=0)
            K = actions_mean.shape[0]
            for _ in range(confg.cl_n_slices)):
                xi = torch.randn(K, ).to(config.device)
                xi /= torch.sqrt((xi**2).sum())
                self.network.zero_grad()
                out = torch.matmul(actions_mean, xi)
                out.backward(retain_graph=True)
                # Update the temporary precision matrix
                for n, p in self.params.items():
                    precision_matrices[n].data += p.grad.data ** 2 / float(len(states))

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(self.device)

        self.network.train()
        return

    def penalty(self):
        loss = 0
        for n, p in self.network.named_parameters():
            _loss = self.precision_matrices[n] * (p - self.means[n]) ** 2
            loss += _loss.sum()
        return loss * self.config.cl_loss_coeff

