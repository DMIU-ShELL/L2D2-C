'''
Source from: https://github.com/RAIVNLab/supsup/blob/master/mnist.ipynb
'''
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

# Subnetwork forward from hidden networks
# Mask derived using Piggyback method (threshold by
# a constant a)
# paper: https://arxiv.org/abs/1801.06519
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores

def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

class MultitaskMaskLinear(nn.Linear):
    def __init__(self, *args, num_tasks=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )
        
        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)
    
    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    GetSubnet.apply(self.scores[j])
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)
        else:
            # Subnet forward pass (given task info in self.task)
            subnet = GetSubnet.apply(self.scores[self.task])
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x


    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def get_mask(self, task):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        return self.scores[task] 
        #return GetSubnet.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks() 
        return

# Subnetwork forward from hidden networks
# Sparse mask (using edge-pop algorithm)
class GetSubnetSparse(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class MultitaskMaskLinearSparse(nn.Linear):
    def __init__(self, *args, num_tasks=1, sparsity=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )
        
        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)

        # sparsity for top k%, edge pop up algorithm
        self.sparsity = sparsity
    
    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    GetSubnetSparse.apply(self.scores[j], self.sparsity)
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)
        else:
            # Subnet forward pass (given task info in self.task)
            subnet = GetSubnetSparse.apply(self.scores[self.task], self.sparsity)
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x


    def __repr__(self):
        return f"MultitaskMaskLinearSparse({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def get_mask(self, task):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        return self.scores[task] 
        #return GetSubnet.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks() 
        return

# Utility functions
def set_model_task(model, task, verbose=True):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse):
            if verbose:
                print(f"=> Set task of {n} to {task}")
            m.task = task

def cache_masks(model):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse):
            print(f"=> Caching mask state for {n}")
            m.cache_masks()

def set_num_tasks_learned(model, num_tasks_learned):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse):
            print(f"=> Setting learned tasks of {n} to {num_tasks_learned}")
            m.num_tasks_learned = num_tasks_learned

def set_alphas(model, alphas, verbose=True):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse):
            if verbose:
                print(f"=> Setting alphas for {n}")
            m.alphas = alphas

def get_mask(model, task):
    mask = {}
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse):
            mask[n] = m.get_mask(task)
    return mask 

def set_mask(model, mask, task):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse):
            m.set_mask(mask[n], task)

# Multitask Model, a simple fully connected model in this case
class MultitaskFC(nn.Module):
    def __init__(self, hidden_size, num_tasks):
        super().__init__()
        self.model = nn.Sequential(
            MultitaskMaskLinear(
                784,
                hidden_size,
                num_tasks=num_tasks,
                bias=False
            ),
            nn.ReLU(),
            MultitaskMaskLinear(
                hidden_size,
                hidden_size,
                num_tasks=num_tasks,
                bias=False
            ),
            nn.ReLU(),
            MultitaskMaskLinear(
                hidden_size,
                100,
                num_tasks=num_tasks,
                bias=False
            )
        )
    
    def forward(self, x):
        return self.model(x.flatten(1))
