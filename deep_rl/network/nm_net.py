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
            self.weight_shape = np.concatenate([self.original_shape, np.ones(1, dtype=np.int64)])
        else:
            self.weight_shape = self.original_shape
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
        #x[x > 0.] = torch.sigmoid(x[x > 0.])
        return x.reshape(*self.original_shape)


