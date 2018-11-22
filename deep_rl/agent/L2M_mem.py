#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *

class L2MMemAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.buffer_update_rate = 0.1;

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        #print 'begin reset'
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        state_buffer = np.asarray(state)
        #print self.network
        #print 'begin episode'
        while True:
            tmp_state = np.asarray(state)
            if steps == 0:
                state_input = np.concatenate((np.array(state_buffer).astype(np.uint8),np.array(state_buffer).astype(np.uint8)),axis=0)
            else:
                state_input = np.concatenate((np.array(state_buffer).astype(np.uint8),tmp_state.astype(np.uint8)),axis=0)
                state_buffer = ((1-self.buffer_update_rate)*state_buffer+self.buffer_update_rate*tmp_state).astype(np.uint8)
            #print np.array(state_buffer).shape
            #print state_nm_signal.shape
            #print np.sum(state_buffer)
            #value = self.network.predict(np.stack([self.config.state_normalizer(state)]), np.stack([state_nm_signal]), True).flatten()
            #print np.sum(tmp_state)
            #print np.sum(self.config.state_normalizer(tmp_state))
            #print np.sum(state_nm_signal)
            #print np.sum(self.config.state_normalizer(state_nm_signal))
            value = self.network.predict(np.stack([self.config.state_normalizer(state_input)]), True).flatten()
            #value = np.zeros(4)
            del tmp_state
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            #print 'call self.task.step(action)'
            next_state, reward, done, _ = self.task.step(action)
            #print 'task step'
            total_reward += reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done), state_input])
                self.total_steps += 1
            steps += 1
            state = next_state
            #print 'learning'
            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency == 0:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals, states_input = experiences
                #print states_buffers.shape
                #print np.sum(states_buffers[10])
                #print np.sum(states_buffers[20])
                states = self.config.state_normalizer(states)
                next_states_buffers = ((1-self.buffer_update_rate)*states_input[:,:4,:,:]+self.buffer_update_rate*states_input[:,4:,:,:]).astype(np.uint8)
                next_states_input = self.config.state_normalizer(np.concatenate((np.array(next_states_buffers).astype(np.uint8),np.asarray(next_states)),axis=1))
                next_states = self.config.state_normalizer(next_states)
                #print np.sum(next_states_nm_signals[:,:4,:,:])
                #print np.sum(next_states_nm_signals[:,4:,:,:])
                #print np.sum(states_nm_signals)
                states_input = self.config.state_normalizer(states_input)
                #print np.sum(states_nm_signals[:,:4,:,:])
                #print np.sum(states_nm_signals[:,4:,:,:])
                q_next = self.target_network.predict(next_states_input, False).detach()
                if self.config.double_q:
                    _, best_actions = self.network.predict(next_states_input).detach().max(1)
                    q_next = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    q_next, _ = q_next.max(1)
                terminals = tensor(terminals)
                rewards = tensor(rewards)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).unsqueeze(1).long()
                q = self.network.predict(states_input, False)
                q = q.gather(1, actions).squeeze(1)
                loss = self.criterion(q, q_next)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            #print 'self evaluate'
            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()
            #print 'chekc is done'
            if done:
                #print 'end eposide'
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps
