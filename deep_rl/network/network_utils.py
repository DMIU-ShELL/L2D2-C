#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import *

class BaseNet:
    def __init__(self):
        pass

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_init_nm(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.fc.weight.data)
    layer.fc.weight.data.mul_(w_scale)
    nn.init.constant_(layer.fc.bias.data, 0)

    nn.init.orthogonal_(layer.nm.fc1.weight.data)
    layer.nm.fc1.weight.data.mul_(w_scale)
    nn.init.constant_(layer.nm.fc1.bias.data, 0)

    nn.init.orthogonal_(layer.nm.fc2.weight.data)
    layer.nm.fc2.weight.data.mul_(w_scale)
    nn.init.constant_(layer.nm.fc2.bias.data, 0)
    return layer

def layer_init_nm_pnn(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.std.weight.data)
    layer.std.weight.data.mul_(w_scale)
    nn.init.constant_(layer.std.bias.data, 0)

    nn.init.orthogonal_(layer.in_nm.weight.data)
    layer.in_nm.weight.data.mul_(w_scale)
    nn.init.constant_(layer.in_nm.bias.data, 0)

    nn.init.orthogonal_(layer.out_nm.weight.data)
    layer.out_nm.weight.data.mul_(w_scale)
    nn.init.constant_(layer.out_nm.bias.data, 0)
    return layer
