#-*- coding: utf-8 -*-
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .network_utils import *

class NMNet(nn.Module):
    def __init__(self, weight_shape, z_dim):
        super(NMNet, self).__init__()
        self.original_shape = np.array(weight_shape)
        if len(self.original_shape) == 1:
            self.weight_shape = np.concatenate([np.ones(1, dtype=np.int64), self.original_shape])
        else:
            self.weight_shape = self.original_shape[::-1]
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, z_dim * int(self.weight_shape[0]))
        self.fc2 = nn.Linear(z_dim, int(self.weight_shape[1]))

    def forward(self, x):
        x = tensor(x)
        # TODO implement a technique if it exist to enforce sparse representations
        # I know ReLU activation enforces some form of sparsity as negative values are
        # zerod out, but I want to enforce a stricter sparsity prior
        x = F.relu(self.fc1(x))
        x = x.reshape(self.weight_shape[0], self.z_dim)
        x = F.relu(self.fc2(x))
        return x.reshape(*self.original_shape)


try:
    from nupic.torch.modules import KWinners
except:
    pass
class NMNetKWinners(nn.Module):
    def __init__(self, weight_shape, z_dim):
        super(NMNetKWinners, self).__init__()
        self.original_shape = np.array(weight_shape)
        if len(self.original_shape) == 1:
            self.weight_shape = np.concatenate([np.ones(1, dtype=np.int64), self.original_shape])
        else:
            self.weight_shape = self.original_shape[::-1]
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, z_dim * int(self.weight_shape[0]))
        self.fc2 = nn.Linear(z_dim, int(self.weight_shape[1]))

        # NOTE fix me, the duty_cycle_period.
        self.act1 = KWinners(n=z_dim*int(self.weight_shape[0]), percent_on=0.2, \
            boost_strength=1.0, duty_cycle_period=200)
        self.act2 = KWinners(n=int(self.weight_shape[1]), percent_on=0.2, \
            boost_strength=1.0, duty_cycle_period=200)

    def forward(self, x):
        x = tensor(x)
        # TODO implement a technique if it exist to enforce sparse representations
        # I know ReLU activation enforces some form of sparsity as negative values are
        # zerod out, but I want to enforce a stricter sparsity prior
        x = self.act1(self.fc1(x))
        x = x.reshape(self.weight_shape[0], self.z_dim)
        #print(x.sum(), x.shape)
        x = self.fc2(x)
        if x.shape[0] < 5 and x.shape[1] < 5:
            # avoid kwinner activation
            x = F.relu(x)
        else:
            x = self.act2(x)
        #print(x.sum(), x.shape)
        return x.reshape(*self.original_shape)

