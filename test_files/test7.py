import torch
from torch import nn
from torch.optim import Adam
import torch.functional as F
import gym

class TransformerActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, embedding_dim, num_heads, num_encoder_layers):
        super(TransformerActorCritic, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, num_heads), num_encoder_layers)

        # Actor network (policy) for predicting action probabilities
        self.actor_fc1 = nn.Linear(embedding_dim, 256)  # Hyperparameter: hidden layer size
        self.actor_fc2 = nn.Linear(256, action_dim)

        # Critic network (value) for estimating state value
        self.critic_fc1 = nn.Linear(embedding_dim, 256)  # Hyperparameter: hidden layer size
        self.critic_fc2 = nn.Linear(256, 1)

    def forward(self, state):
        # Encode the state (assuming state is a single tensor for the current state)
        state = state.unsqueeze(0)  # Add batch dimension if state is a single tensor
        encoded_state = self.encoder(state)

        # Actor network
        actor_out = F.relu(self.actor_fc1(encoded_state[:, 0, :]))  # Process first token output
        action_probs = F.softmax(self.actor_fc2(actor_out), dim=1)

        # Critic network
        critic_out = F.relu(self.critic_fc1(encoded_state[:, 0, :]))
        state_value = self.critic_fc2(critic_out)

        return action_probs, state_value

def train_actor_critic(env, model, optimizer, gamma=0.99, num_episodes=1000):
  # Task change detection (simplified example)
  last_action = None
  task_changed = False

  for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
      # Convert state to torch tensor
      state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

      # Get action probabilities and state value
      action_probs, state_value = model(state_tensor)

      # Sample action based on probabilities (exploration vs exploitation strategy)
      action = torch.multinomial(action_probs, 1).item()

      # Perform action and observe next state, reward, and done flag
      next_state, reward, done, _ = env.step(action)
      total_reward += reward

      # Task change detection (check for significant action change)
      if last_action is not None and abs(last_action - action) > threshold:
        task_changed = True

      # Advantage calculation (replace with your chosen advantage function)
      advantage = reward + gamma * model(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))[1].item() - state_value.item()

      # Policy loss (negative log-likelihood of chosen action)
      policy_loss = -torch.log(action_probs[0][action]) * advantage

      # Value loss (mean squared error between estimated and actual reward)
      value_loss = F.mse_loss(state_value, torch.tensor(reward + gamma * model(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))[1].item()).unsqueeze(0))

      # Backpropagate and update model parameters
      optimizer.zero_grad()
      loss = policy_loss + value_loss
      loss.backward()
      optimizer.step()

      # Update last action and state for next iteration
      last_action = action
      state = next_state

      if done:
        print(f"Episode {episode+1} finished with total reward {total_reward}. Task changed: {task_changed}")
        task_changed = False  # Reset task change flag

if __name__ == "__main__":
  env_names = ""
  env = gym.make("CartPole-v1")
  model = TransformerActorCritic(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, embedding_dim=128, num_heads=4, num_encoder_layers=2)
  optimizer = Adam(model.parameters(), lr=0.001)
  train_actor_critic(env, model, optimizer)
  env.close()
