#-*- coding: utf-8 -*-
import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
import torch.nn as nn

class Hypernet_(nn.Module):
    def __init__(self, weight_shape, z_dim=64):
        super(Hypernet_, self).__init__()
        self.weight_shape = weight_shape
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, z_dim * weight_shape[0] )
        self.fc2 = nn.Linear(z_dim, weight_shape[1] )

    def forward(self, x):
        # TODO implement a technique if it exist to enforce sparse representations
        # I know ReLU activation enforces some form of sparsity as negative values are
        # zerod out, but I want to enforce a stricter sparsity prior
        x = F.relu(self.fc1(x))
        x = x.reshape(self.weight_shape[0], self.z_dim)
        x = F.relu(self.fc2(x))
        return x

class NMLinear(Module):
    def __init__(self, in_features, out_features, z_dim):
        self.in_features = in_features
        self.out_features = out_features
        self.z_dim = z_dim
        self.fc = nn.Linear(in_features, out_features)
        self.nm = Hypernet_((in_features, out_features), z_dim)

    def forward(self, x, task_label):
        x = self.fc(x)
        impt_params = self.nm(task_label)
        return x, impt_params

class NMLinearPNN(Module):
    def __init__(self, in_features, out_features, nm_features, bias=True, gate='soft'):
        super(NMLinearPNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nm_features = nm_features
        self.in_nm_act = F.relu # NOTE hardcoded activation function
        self.out_nm_act = torch.tanh # NOTE hardcoded activation function

        self.std = nn.Linear(in_features, out_features, bias=bias)
        self.in_nm = nn.Linear(in_features, nm_features, bias=bias)
        self.out_nm = nn.Linear(nm_features, out_features, bias=bias)
        self.gate = gate

    def forward(self, data, params=None):
        output = self.std(data)
        mod_features = self.in_nm_act(self.in_nm(data))
        sign_ = self.out_nm_act(self.out_nm(mod_features))
        if self.gate == 'hard':
            sign_ = torch.sign(sign_)
            sign_[sign_ == 0.] = 1. # a zero value should have sign of 1. and not 0.
        output *= sign_
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, nm_features={}'.format(self.in_features,\
                self.out_features, self.nm_features)
