import matplotlib.pyplot as plt
import numpy as np

# Function to map the reward
def linear_function(reward):
    return 0.5 + 0.5 * reward

# Generate input rewards from 0 to 1
input_rewards = np.linspace(0, 1, 100)
# Map these rewards to the new range
mapped_rewards = linear_function(input_rewards)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(input_rewards, mapped_rewards, label='Mapped Reward')
plt.xlabel('Input Reward (0 to 1)')
plt.ylabel('Normalized Reward Mapped Weights (Beta parameters) (0.5 to 1)')
plt.title('Mapping of Input Reward to Mapped Reward')
plt.legend()
plt.grid(True)
plt.savefig('reward_mapping.pdf', dpi=300)





# Define the piecewise linear-nonlinear function
def sigmoid_function(x, x0=0.5, k=10):
    return 0.5 + 0.5*(1 / (1 + np.exp(-14 * (x-0.5)))) * ((1-0.001) + 0.001)

# Generate input values
x_values = np.linspace(0, 1, 400)
y_values = sigmoid_function(x_values)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='Piecewise Linear-Nonlinear Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Piecewise Linear-Nonlinear Function')
plt.legend()
plt.grid(True)
plt.savefig('piecewise.pdf', dpi=300)
