import torch
import torch.nn.functional as F

# Define the dictionary with a larger number of entries and larger tensors
data_dict = {}

# Add more entries with larger tensors
for i in range(0, 100):
    tensor_size = 3  # Adjust the size of the tensors as needed
    random_tensor = torch.rand(tensor_size, dtype=torch.float32)  # Generate a random tensor
    random_reward = torch.rand(1, dtype=torch.float32)  # Generate a random reward

    data_dict[i] = {'task_emb': random_tensor, 'reward': random_reward.item()}

data_dict = {0: {'task_emb': torch.tensor([0.1, 0.1, 0.1]), 'reward': 1.0}}

# Tensor to search for
target_tensor = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
target_reward = 0.40  # Adjust the target reward as needed

# Extract embeddings and rewards from the data_dict
embeddings = []
rewards = []

for value in data_dict.values():
    if 'task_emb' in value:
        known_embedding = value['task_emb']
        known_reward = value['reward']

        # Apply the condition: 0.9 * known_reward > target_reward
        if 0.9 * known_reward > target_reward:
            embeddings.append(known_embedding)  # Use the PyTorch tensor directly
            rewards.append(known_reward)

if embeddings:
    # Convert to a list of PyTorch tensors
    embeddings = torch.stack(embeddings)
    #rewards = torch.stack(rewards)
    

    # Calculate cosine similarities using PyTorch
    print(embeddings, target_tensor, target_tensor.unsqueeze(0))
    similarities = F.cosine_similarity(embeddings, target_tensor.unsqueeze(0))
    print(type(similarities))

    print('embeddings = ', embeddings)
    print('similarities =', similarities)

    # Find the index of the closest entry
    closest_index = torch.argmax(similarities).item()
    print(torch.argmax(similarities))

    # Get the corresponding key
    closest_key = list(data_dict.keys())[closest_index]

    # Get the closest similarity
    closest_similarity = torch.min(similarities).item()
    print('hello', similarities[closest_index], type(similarities[closest_index]))
    print(torch.min(similarities))

    # Define a threshold for cosine similarity (adjust as needed)
    cosine_similarity_threshold = 0.9  # Example threshold

    # Check if the closest similarity is above the threshold
    if closest_similarity >= cosine_similarity_threshold:
        # Print the closest key and distance
        print(f"The key with the closest 'emb' satisfying the condition is {closest_key} with a similarity of {closest_similarity}")
