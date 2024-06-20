import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import procgen

class convnet(nn.Module):
    def __init__(self):
        super(convnet, self).__init__()  # Inherits from nn.Module
        in_channels = 3  # Assuming RGB image from ImgObsWrapper
        feature_dim = 512
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate output size of conv layers (assuming no padding)
        #out_size = self._get_conv_out_size((in_channels, 84, 84), self.conv1)
        #out_size = self._get_conv_out_size(out_size, self.conv2)
        #out_size = self._get_conv_out_size(out_size, self.conv3)

        self.fc = nn.Linear(64*7*7, feature_dim)  # Linear layer size adjusted

        self.gate = F.relu  # ReLU activation function

    def _get_conv_out_size(self, input_shape, conv_layer):
        # Helper function to calculate output size of convolutional layers
        # Considering stride and kernel size
        out_w = (input_shape[0] - conv_layer.kernel_size[0] + 2 * conv_layer.padding[0]) // conv_layer.stride[0]
        out_h = (input_shape[1] - conv_layer.kernel_size[1] + 2 * conv_layer.padding[1]) // conv_layer.stride[1]
        return (out_w, out_h)

    def forward(self, x):
        y = self.gate(self.conv1(x))
        y = self.gate(self.conv2(y))
        y = self.gate(self.conv3(y))
        y = y.view(y.shape[0], -1)  # Flatten the output
        y = self.gate(self.fc(y))
        return y


if __name__ == '__main__':
    env = gym.make('procgen:procgen-coinrun-v0')

    obs = env.reset()
    #obs = torch.tensor(obs).unsqueeze(0)  # Convert observation to PyTorch tensor with batch dimension

    network = convnet()

    # Assuming obs is your observation data
    print(obs.shape, len(obs))

    print(obs[0].shape)

    for i in range(0, 100):
        obs, rew, done, info = env.step(env.action_space.sample())
        obs = obs / 255.0  # Normalize between 0.0 and 1.0 (optional)
        action = network.forward(obs)

    # Training would typically involve using the action prediction and the environment reward
    # to update the network weights through an optimizer
