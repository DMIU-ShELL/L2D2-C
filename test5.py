import torch

def add_embedding(self, embedding, reward):
       self.embeddings.append(embedding)
       self.rewards.append(reward)

def calculate_weighted_average(self):
    if not self.embeddings or not self.rewards:
        raise ValueError("No embeddings or rewards added yet.")

    # Convert rewards to PyTorch tensor for easier manipulation
    rewards_tensor = torch.tensor(self.rewards, dtype=torch.float)

    # Add a small epsilon value to prevent division by zero
    epsilon = torch.tensor(1e-8)
    total_reward = torch.sum(rewards_tensor) + epsilon

    # Normalize rewards to sum up to 1
    normalized_rewards = rewards_tensor / total_reward

    # Calculate weighted average
    weighted_sum = torch.zeros_like(self.embeddings[0], dtype=torch.float)
    for embedding, reward in zip(self.embeddings, normalized_rewards):
        weighted_sum += reward * embedding
    weighted_average = weighted_sum / len(self.embeddings)

    return weighted_average


if __name__ == '__main__':
    tensors = [
         torch.tensor([0., 0., 0.]),
         torch.tensor([-0.0754,  0.0310,  0.0169]),
         torch.tensor([-0.0719,  0.0350,  0.0209]),
         torch.tensor([0.0137, -0.1382, -0.0741])
    ]


    for tensor in tensors:
        print()