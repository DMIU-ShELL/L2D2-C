'''
Adapted and extended from: https://github.com/RAIVNLab/supsup/blob/master/mnist.ipynb
'''
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import copy
from colorama import Fore
import traceback
from ...utils.config import Config

# Subnetwork forward from hidden networks
# Mask derived using Piggyback method (threshold by
# a constant a)
# paper: https://arxiv.org/abs/1801.06519
class GetSubnetDiscrete(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

class GetSubnetContinuous(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float() * scores

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

# may not work as well as the alternative above (which is the default choice)
class GetSubnetContinuousV2:
    @staticmethod
    def apply(scores, a=0):
        return (scores >= a).float() * scores

def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores

def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

NEW_MASK_RANDOM = 'random'
NEW_MASK_LINEAR_COMB = 'linear_comb'
class MultitaskMaskLinear(nn.Linear):
    def __init__(self, *args, discrete=True, num_tasks=1, new_mask_type=NEW_MASK_RANDOM, bias=False, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
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

        self.task = -1
        self.num_tasks_learned = 0
        self.new_mask_type = new_mask_type
        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
            self._forward_mask = self._forward_mask_linear_comb
        else:
            self.betas = None
            self._forward_mask = self._forward_mask_normal

        # subnet class
        self._subnet_class = GetSubnetDiscrete if discrete else GetSubnetContinuous

        # to initialize/register the stacked module buffer.
        self.cache_masks()
    
    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self._subnet_class.apply(self.scores[j])
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            raise ValueError('`self.task` should be set to >= 0')
            '''# Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)'''
        else:
            # Subnet forward pass (given task info in self.task)
            #subnet = self._subnet_class.apply(self.scores[self.task])
            subnet = self._forward_mask()#NOTE Here is where we crash !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x

    def _forward_mask_normal(self):
        return self._subnet_class.apply(self.scores[self.task])

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.task]
        # NOTE comment/uncomment the code block below to disble/enable the use of consolidated masks
        # in PPO_agent (in trask_train_end(...)) also comment/uncomment `consolidate_mask` function.
        if self.task < self.num_tasks_learned:
            # this is a task that has been seen before (with established/trained mask).
            # fetch mask and use (either for eval or to continue training).
            return self._subnet_class.apply(_subnet)

        # otherwise, this is a new task. check if the first task
        if self.task == 0:
            # this is the first task to train. no previous task mask to linearly combine.
            return self._subnet_class.apply(_subnet)

        # otherwise, a new task and it is not the first task. combine task mask with
        # masks from previous tasks.
        # note: should not update scores/masks from previous tasks. only update their coeffs/betas
        _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        assert len(_subnets) > 0, 'an error occured'
        _betas = self.betas[self.task, 0:self.task+1]
        _betas = torch.softmax(_betas, dim=-1)
        _subnets.append(_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        return self._subnet_class.apply(_subnet_linear_comb)

    @torch.no_grad()
    def consolidate_mask(self):
        print('USING THE WRONG CONSOLIDATION METHOD!!!!!!!!!!!!!!!!!!!', flush=True)
        if self.new_mask_type == NEW_MASK_RANDOM: return
        if self.task <= 0: return
        if self.task < self.num_tasks_learned:
            # re-visiting a task that has been previously learnt (no need to consolidate)
            # which should not get here though, because this secanrio should have been caught
            # task_train_end(...) method in supermask_policy.py class.
            # assert False, 'sanity check'
            return
        _subnet = self.scores[self.task]
        _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        assert len(_subnets) > 0, 'an error occured'
        _betas = self.betas[self.task, 0:self.task+1]
        _betas = torch.softmax(_betas, dim=-1)
        _subnets.append(_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        self.scores[self.task].data = _subnet_linear_comb.data
        return
        
    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            return self.scores[task]
        else:
            return self._subnet_class.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks() 
        return

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            if task > 0:
                k = task + 1
                self.betas.data[task, 0:k] = 1. / k
                #print(self.betas)

class ComposeMultitaskMaskLinear(nn.Linear):
    def __init__(self, *args, discrete=True, num_tasks=1, max_community_masks=5, seed=1, new_mask_type=NEW_MASK_RANDOM, bias=False, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self.num_tasks = num_tasks
        
        # Pre-generate the random masks for each task using mask_init() while keeping the backbone network the same
        #torch.manual_seed(seed)
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )

        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)

        self.task = -1
        self.num_tasks_learned = 0
        self.new_mask_type = new_mask_type
        
        self.comm_masks = []
        self.num_comm_masks = len(self.comm_masks)
        self.k = max_community_masks

        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(num_tasks, self.k + num_tasks).type(torch.float32))
            self._forward_mask = self._forward_mask_linear_comb
        else:
            self.betas = None
            self._forward_mask = self._forward_mask_normal

        # subnet class
        self._subnet_class = GetSubnetDiscrete if discrete else GetSubnetContinuous

        # to initialize/register the stacked module buffer.
        self.cache_masks()
    
    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self._subnet_class.apply(self.scores[j])
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            raise ValueError('`self.task` should be set to >= 0')
            '''# Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)'''
        else:
            # Subnet forward pass (given task info in self.task)
            #subnet = self._subnet_class.apply(self.scores[self.task])
            subnet = self._forward_mask()#NOTE Here is where we crash !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x

    def _forward_mask_normal(self):
        return self._subnet_class.apply(self.scores[self.task])

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.task]    # Get our current task training subnet

        if self.task == 0 or self.task < self.num_tasks_learned:
            if len(self.comm_masks) == 0:
                return self._subnet_class.apply(_subnet)
            
            _subnets = [_subnet]
            _subnets.extend([c.detach() for c in self.comm_masks])

            _betas = self.betas[self.task, 0:self.task+1+self.num_comm_masks]    # Get betas for community masks and subnet
            _betas = torch.softmax(_betas, dim=-1)  # Apply softmax to the betas

            # Perform linear combinantions
            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]    
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            return self._subnet_class.apply(_subnet_linear_comb)
        
        else:
            _subnets = [self.scores[idx].detach() for idx in range(self.task)]
            assert len(_subnets) > 0, 'an error occured'
            _subnets.append(_subnet)

            if len(self.comm_masks) > 0: _subnets.extend([c.detach() for c in self.comm_masks])
            # This gives us _subnets = [ LLmasks..., current subnet..., community masks... ]


            _betas = self.betas[self.task, 0:self.task+1+self.num_comm_masks]    # Get the row from 2D betas for this task up to the number of community masks (k)
            _betas = torch.softmax(_betas, dim=-1)
            assert len(_betas) == len(_subnets), 'an error occured'

            # Perform linear combinantions
            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]    
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            return self._subnet_class.apply(_subnet_linear_comb)
    
    @torch.no_grad()
    def consolidate_mask(self):
        """
        Consolidate all the masks (LL, community and current training subnet) at end of training on task
        """
        if self.new_mask_type == NEW_MASK_RANDOM: return
        if self.task < self.num_tasks_learned: return
        if self.task <= 0 and len(self.comm_masks) == 0: return # If there are no LL masks (because we are on the first task) and there are no community masks, then there is no need for consolidation.
        if self.task > 0:  # Not on the first task so we can expect LL masks
            _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        
        _subnet = self.scores[self.task]
        _subnets.append(_subnet)
        _subnets.extend([c.detach() for c in self.comm_masks])
        assert len(_subnets) > 0, 'an error occured'
        
        _betas = self.betas[self.task, 0:self.task+1+self.num_comm_masks]
        _betas = torch.softmax(_betas, dim=-1)
        assert len(_betas) == len(_subnets), 'an error ocurred'

        
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        self.scores[self.task].data = _subnet_linear_comb.data
        return

    @torch.no_grad()
    def consolidate_comm_mask(self):
        if self.new_mask_type == NEW_MASK_RANDOM: 
            print(0)
            return
        if self.task < self.num_tasks_learned: 
            print(1)
            return
        if len(self.comm_masks) == 0: 
            print(2)
            return # If there are no community masks, then there is no need for consolidation.
        if self.betas is None:
            print(3)
            return

        print(4)
        _subnet = self.scores[self.task]
        _subnets = [_subnet]
        _subnets.extend([c.detach() for c in self.comm_masks])
        assert len(_subnets) > 0, 'an error occured'
        
        _betas = self.betas[self.task, self.task:self.task+1+self.num_comm_masks]
        print(f'Betas before softmax: {_betas}')
        _betas = torch.softmax(_betas, dim=-1)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        print(f'Betas after softmax: {_betas}')

        
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        self.scores[self.task].data = _subnet_linear_comb.data
        return
    
    @torch.no_grad()
    def update_comm_masks(self, masks):
        print('Consolidating existing comm masks')
        self.consolidate_comm_mask() # Consolidate the current community masks only NOTE: Replace this with consolidate_mask() if we want to consolidate all knowledge
        self.cache_masks()  # NOTE: I'm not sure if we need to do this, but following the original examples I've put it in.
        
        print('Replacing with new comm masks')
        self.comm_masks = [mask.to(Config.DEVICE) for mask in masks] # Update to the new community masks
        self.num_comm_masks = len(self.comm_masks)  # Update number of comm masks
        
        # Reset the betas for the community masks
        #if self.new_mask_type == NEW_MASK_LINEAR_COMB:
        #    self.betas.data[self.task, self.task+1:self.task+1+self.num_comm_masks] = 1/(2*self.num_comm_masks)

        print(f'Betas: {self.betas}')

    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            return self.scores[task]
        else:
            return self._subnet_class.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks() 
        return

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            if task > 0:
                self.betas.data[task, 0:self.task+1+self.k] = 1. / self.task + 1 + self.k
                
class CompBLC_MultitaskMaskLinear(nn.Linear):
    def __init__(self, *args, discrete=True, num_tasks=1, max_community_masks=5, seed=1, new_mask_type=NEW_MASK_RANDOM, bias=False, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self.num_tasks = num_tasks
        
        # Pre-generate the random masks for each task using mask_init() while keeping the backbone network the same
        #torch.manual_seed(seed)
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )

        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)

        self.task = -1
        self.num_tasks_learned = 0
        self.new_mask_type = new_mask_type
        
        self.comm_masks = []
        self.num_comm_masks = len(self.comm_masks)
        self.k = max_community_masks

        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(num_tasks, self.k + num_tasks).type(torch.float32))
            self._forward_mask = self._forward_mask_linear_comb
        else:
            self.betas = None
            self._forward_mask = self._forward_mask_normal

        # subnet class
        self._subnet_class = GetSubnetDiscrete if discrete else GetSubnetContinuous

        # to initialize/register the stacked module buffer.
        self.cache_masks()
    
    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self._subnet_class.apply(self.scores[j])
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            raise ValueError('`self.task` should be set to >= 0')
            '''# Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)'''
        else:
            # Subnet forward pass (given task info in self.task)
            #subnet = self._subnet_class.apply(self.scores[self.task])
            subnet = self._forward_mask()#NOTE Here is where we crash !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x

    def _forward_mask_normal(self):
        return self._subnet_class.apply(self.scores[self.task])

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.task]    # Get our current task training subnet

        if self.task == 0 or self.task < self.num_tasks_learned:
            if len(self.comm_masks) == 0:
                return self._subnet_class.apply(_subnet)
            
            _subnets = [_subnet]
            _subnets.extend([c.detach() for c in self.comm_masks])

            _betas = self.betas[self.task, 0:self.task+1+self.num_comm_masks]    # Get betas for community masks and subnet
            _betas = torch.softmax(_betas, dim=-1)  # Apply softmax to the betas

            # Perform linear combinantions
            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]    
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            return self._subnet_class.apply(_subnet_linear_comb)
        
        else:
            _subnets = [self.scores[idx].detach() for idx in range(self.task)]
            assert len(_subnets) > 0, 'an error occured'
            _subnets.append(_subnet)

            if len(self.comm_masks) > 0: _subnets.extend([c.detach() for c in self.comm_masks])
            # This gives us _subnets = [ LLmasks..., current subnet..., community masks... ]


            _betas = self.betas[self.task, 0:self.task+1+self.num_comm_masks]    # Get the row from 2D betas for this task up to the number of community masks (k)
            _betas = torch.softmax(_betas, dim=1)
            assert len(_betas) == len(_subnets), 'an error occured'

            # Perform linear combinantions
            _subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]    
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            return self._subnet_class.apply(_subnet_linear_comb)
    
    @torch.no_grad()
    def consolidate_mask(self):
        """
        Consolidate all the masks (LL, community and current training subnet) at end of training on task
        """
        if self.new_mask_type == NEW_MASK_RANDOM: return
        if self.task < self.num_tasks_learned: return
        if self.task <= 0 and len(self.comm_masks) == 0: return # If there are no LL masks (because we are on the first task) and there are no community masks, then there is no need for consolidation.
        if self.task > 0:  # Not on the first task so we can expect LL masks
            _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        
        _subnet = self.scores[self.task]
        _subnets.append(_subnet)
        _subnets.extend([c.detach() for c in self.comm_masks])
        assert len(_subnets) > 0, 'an error occured'
        
        _betas = self.betas[self.task, 0:self.task+1+self.num_comm_masks]
        _betas = torch.softmax(_betas, dim=-1)
        assert len(_betas) == len(_subnets), 'an error ocurred'

        
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        self.scores[self.task].data = _subnet_linear_comb.data
        return

    @torch.no_grad()
    def consolidate_comm_mask(self):
        if self.new_mask_type == NEW_MASK_RANDOM: 
            print(0)
            return
        if self.task < self.num_tasks_learned: 
            print(1)
            return
        if len(self.comm_masks) == 0: 
            print(2)
            return # If there are no community masks, then there is no need for consolidation.
        if self.betas is None: 
            print(3)
            return
        
        print(4)

        _subnet = self.scores[self.task]
        _subnets = [_subnet]
        _subnets.extend([c.detach() for c in self.comm_masks])
        assert len(_subnets) > 0, 'an error occured'
        
        _betas = self.betas[self.task, self.task:self.task+1+self.num_comm_masks]
        print(f'Betas before softmax: {_betas}')
        _betas = torch.softmax(_betas, dim=-1)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        print(f'Betas after softmax: {_betas}')

        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        self.scores[self.task].data = _subnet_linear_comb.data
        return
    
    @torch.no_grad()
    def update_comm_masks(self, masks):
        print('Consolidating existing comm masks')
        self.consolidate_comm_mask() # Consolidate the current community masks only NOTE: Replace this with consolidate_mask() if we want to consolidate all knowledge
        self.cache_masks()  # NOTE: I'm not sure if we need to do this, but following the original examples I've put it in.
        
        print('Replacing with new comm masks')
        self.comm_masks = [mask.to(Config.DEVICE) for mask in masks] # Update to the new community masks
        self.num_comm_masks = len(self.comm_masks)  # Update number of comm masks
        
        print(self.betas)
        # Reset the betas for the community masks
        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            # Undo log operation
            self.betas.data[self.task, 0:self.task+1+self.num_comm_masks] = torch.exp(self.betas.data[self.task, 0:self.task+1+self.num_comm_masks])
            
            # Reset the community betas using (BLC) technique based off of the current value of the beta for the current subnet
            # NOTE: This method will ensure that the total weighting adds up to 1 every time. It should also adjust the weightings
            #       given to community masks, as the agents local policy gets better and better, putting more weight on the local subnet.
            threshold = self.betas.data[self.task, self.task:self.task+1] # Get the value of the beta parameter for the current subnet
            remainder = 1. - threshold  # Calculate remaining weighting
            self.betas.data[self.task, self.task+1:self.task+1+self.num_comm_masks] = remainder / self.num_comm_masks

            # Reset the beta parameter initialization for the number of community masks
            #self.betas.data[self.task, self.task+1:self.task+1+self.num_comm_masks] = 1/(2*self.num_comm_masks)
            
            # Redo the log operation
            self.betas.data[self.task, 0:self.task+1+self.num_comm_masks] = torch.log(self.betas.data[self.task, 0:self.task+1+self.num_comm_masks])
        
        print(f'Betas: {self.betas}')

    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            return self.scores[task]
        else:
            return self._subnet_class.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks() 
        return

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        print('IN SET_TASK()')
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            k = task + 1
            self.betas.data[task, 0:k] = 1 / (2*k)
            self.betas.data[task, k:k+self.k] = 1 / (2*self.k)
            self.betas.data[task, 0:k+self.k] = torch.log(self.betas.data[task, 0:k+self.k])

        print(self.betas)
                
      

# Subnetwork forward from hidden networks
# Sparse mask (using edge-pop algorithm)
class GetSubnetSparseDiscrete(autograd.Function):
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

class GetSubnetSparseContinuous(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

# may not work as well as the alternative above (which is the default choice)
class GetSubnetSparseContinuousV2:
    @staticmethod
    def apply(scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0

        return out

class MultitaskMaskLinearSparse(nn.Linear):
    def __init__(self, *args, discrete=True, num_tasks=1, sparsity=0.5, \
        new_mask_type=NEW_MASK_RANDOM, bias=False, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
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
    
        self.task = -1
        self.num_tasks_learned = 0
        self.new_mask_type = new_mask_type
        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
            self._forward_mask = self._forward_mask_linear_comb
        else:
            self.betas = None
            self._forward_mask = self._forward_mask_normal

        # subnet class
        self._subnet_class = GetSubnetSparseDiscrete if discrete else GetSubnetSparseContinuous

        # to initialize/register the stacked module buffer.
        self.cache_masks()

    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self._subnet_class.apply(self.scores[j], self.sparsity)
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
            #subnet = self._subnet_class.apply(self.scores[self.task], self.sparsity)
            subnet = self._forward_mask()
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x

    def _forward_mask_normal(self):
        return self._subnet_class.apply(self.scores[self.task], self.sparsity)

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.task]
        # NOTE comment/uncomment the code block below to disble/enable the use of consolidated masks
        # in PPO_agent (in trask_train_end(...)) also comment/uncomment `consolidate_mask` function.
        if self.task < self.num_tasks_learned:
            # this is a task that has been seen before (with established/trained mask).
            # fetch mask and use (either for eval or to continue training).
            return self._subnet_class.apply(_subnet, self.sparsity)

        # otherwise, this is a new task. check if the first task
        if self.task == 0:
            # this is the first task to train. no previous task mask to linearly combine.
            return self._subnet_class.apply(_subnet, self.sparsity)

        # otherwise, a new task and it is not the first task. combine task mask with
        # masks from previous tasks.
        # note: should not update scores/masks from previous tasks. only update their coeffs/betas
        _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        assert len(_subnets) > 0, 'an error occured'
        _betas = self.betas[self.task, 0:self.task+1]
        _betas = torch.softmax(_betas, dim=-1)
        _subnets.append(_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        return self._subnet_class.apply(_subnet_linear_comb, self.sparsity)

    @torch.no_grad()
    def consolidate_mask(self):
        if self.task <= 0 or self.new_mask_type == NEW_MASK_RANDOM:
            return
        _subnet = self.scores[self.task]
        _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        assert len(_subnets) > 0, 'an error occured'
        _betas = self.betas[self.task, 0:self.task+1]
        _betas = torch.softmax(_betas, dim=-1)
        _subnets.append(_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        self.scores[self.task].data = _subnet_linear_comb.data
        return

    def __repr__(self):
        return f"MultitaskMaskLinearSparse({self.in_dims}, {self.out_dims})"

    @torch.no_grad()
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            return self.scores[task]
        else:
            return self._subnet_class.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks() 
        return

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            if task > 0:
                k = task + 1
                self.betas.data[task, 0:k] = 1. / k

# Utility functions
def set_model_task(model, task, verbose=True, new_task=False):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or \
            isinstance(m, MultitaskMaskConv2d) or isinstance(m, MultitaskMaskConv2dSparse) or \
                isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            if verbose:
                print(f"=> Set task of {n} to {task}")
            m.set_task(task, new_task)

def cache_masks(model, verbose=False):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or \
            isinstance(m, MultitaskMaskConv2d) or isinstance(m, MultitaskMaskConv2dSparse) or \
                isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            if verbose:
                print(f"=> Caching mask state for {n}")
            m.cache_masks()

def set_num_tasks_learned(model, num_tasks_learned):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or \
            isinstance(m, MultitaskMaskConv2d) or isinstance(m, MultitaskMaskConv2dSparse) or \
                isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            print(f"=> Setting learned tasks of {n} to {num_tasks_learned}")
            m.num_tasks_learned = num_tasks_learned

def set_alphas(model, alphas, verbose=False):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or \
            isinstance(m, MultitaskMaskConv2d) or isinstance(m, MultitaskMaskConv2dSparse) or \
                isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            if verbose:
                print(f"=> Setting alphas for {n}")
            m.alphas = alphas

def get_mask(model, task, raw_score=True):
    mask = {}
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or \
            isinstance(m, MultitaskMaskConv2d) or isinstance(m, MultitaskMaskConv2dSparse) or \
                isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            mask[n] = m.get_mask(task, raw_score)
    return mask 

def set_mask(model, mask, task):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or \
                isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            m.set_mask(mask[n], task)

def consolidate_mask(model):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            m.consolidate_mask()

def erase_masks(model, device):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear) or isinstance(m, MultitaskMaskLinearSparse) or \
                isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            num_tasks = m.num_tasks
            # m.scores = nn.ParameterList(
            #     [
            #         nn.Parameter(mask_init(m))
            #         for _ in range(num_tasks)
            #     ]
            # ).to("cuda:0")

            for i in range(num_tasks):
                m.scores[i].data = mask_init(m).to(device)
                m.num_tasks_learned = 0
                m.task = 0

def consolidate_comm_mask(model):
    for n, m in model.named_modules():
        if isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            m.consolidate_comm_masks()

def update_comm_masks(model, masks):
    for n, m in model.named_modules():
        if isinstance(m, ComposeMultitaskMaskLinear) or isinstance(m, CompBLC_MultitaskMaskLinear):
            m.update_comm_masks([mask[n] for mask in masks])


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

# adapted from: https://github.com/RAIVNLab/supsup/blob/master/models/modules.py
class MultitaskMaskConv2d(nn.Conv2d):
    def __init__(self, *args, discrete=True, num_tasks=1, new_mask_type=NEW_MASK_RANDOM, \
        bias=False, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList(
            [nn.Parameter(mask_init(self)) for _ in range(num_tasks)]
        )

        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)

        self.task = -1
        self.num_tasks_learned = 0
        self.new_mask_type = new_mask_type
        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
            self._forward_mask = self._forward_mask_linear_comb
        else:
            self.betas = None
            self._forward_mask = self._forward_mask_normal

        # subnet class
        self._subnet_class = GetSubnetDiscrete if discrete else GetSubnetContinuous

        # to initialize/register the stacked module buffer.
        self.cache_masks()

    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self._subnet_class.apply(self.scores[j])
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        if self.task < 0:
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)
        else:
            #subnet = GetSubnet.apply(self.scores[self.task])
            subnet = self._forward_mask()

        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def _forward_mask_normal(self):
        return self._subnet_class.apply(self.scores[self.task])

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.task]
        # NOTE comment/uncomment the code block below to disble/enable the use of consolidated masks
        # in PPO_agent (in trask_train_end(...)) also comment/uncomment `consolidate_mask` function.
        if self.task < self.num_tasks_learned:
            # this is a task that has been seen before (with established/trained mask).
            # fetch mask and use (either for eval or to continue training).
            return self._subnet_class.apply(_subnet)

        # otherwise, this is a new task. check if the first task
        if self.task == 0:
            # this is the first task to train. no previous task mask to linearly combine.
            return self._subnet_class.apply(_subnet)

        # otherwise, a new task and it is not the first task. combine task mask with
        # masks from previous tasks.
        # note: should not update scores/masks from previous tasks. only update their coeffs/betas
        _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        assert len(_subnets) > 0, 'an error occured'
        _betas = self.betas[self.task, 0:self.task+1]
        _betas = torch.softmax(_betas, dim=-1)
        _subnets.append(_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        return self._subnet_class.apply(_subnet_linear_comb)

    @torch.no_grad()
    def consolidate_mask(self):
        self.scores[self.task].data = self._consolidate_mask()
        return

    @torch.no_grad()
    def _consolidate_mask(self):
        # catch scenarios where consolidation of mask is NOT needed.
        if self.new_mask_type == NEW_MASK_RANDOM: return
        if self.task <= 0: return
        if self.task < self.num_tasks_learned:
            # re-visiting a task that has been previously learnt (no need to consolidate)
            # which should not get here though, because this secanrio should have been caught
            # task_train_end(...) method in supermask_policy.py class.
            # assert False, 'sanity check'
            return

        _subnet = self.scores[self.task]
        _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        assert len(_subnets) > 0, 'an error occured'
        _betas = self.betas[self.task, 0:self.task+1]
        _betas = torch.softmax(_betas, dim=-1)
        _subnets.append(_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        return _subnet_linear_comb.data
        

    def __repr__(self):
        return f"MultitaskMaskConv2d({self.in_channels}, {self.out_channels})"

    @torch.no_grad()
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            return self.scores[task]
        else:
            return self._subnet_class.apply(self.scores[task])

    '''@torch.no_grad() 
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            # Check if we are using MASK RI or MASK LC
            if self.new_mask_type == NEW_MASK_RANDOM:
                return self.scores[task]

            # MASK LC
            else:
                # If mask requested is being trained on then
                if task < self.task:
                    return self.scores[task]

                elif task == self.task:
                    # Assuming two agents are combining 
                    return self.consolidate_mask()

                else:
                    # Requesting a mask where the agent hasn't trained for the task
                    raise ValueError('sanity check')
                    # return self.scores[task]  # Return random initialised mask
        else:
            return self._subnet_class.apply(self.scores[task])'''

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks()
        return

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            if task > 0:
                k = task + 1
                self.betas.data[task, 0:k] = 1. / k

# adapted from: https://github.com/RAIVNLab/supsup/blob/master/models/modules.py
class MultitaskMaskConv2dSparse(nn.Conv2d):
    def __init__(self, *args, discrete=True, num_tasks=1, sparsity=0.5, \
        new_mask_type=NEW_MASK_RANDOM, bias=False, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )
        self.weight.requires_grad = False
        signed_constant(self)

        # sparsity for top k%, edge pop up algorithm
        self.sparsity = sparsity

        self.task = -1
        self.num_tasks_learned = 0
        self.new_mask_type = new_mask_type
        if self.new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(num_tasks, num_tasks).type(torch.float32))
            self._forward_mask = self._forward_mask_linear_comb
        else:
            self.betas = None
            self._forward_mask = self._forward_mask_normal

        # subnet class
        self._subnet_class = GetSubnetSparseDiscrete if discrete else GetSubnetSparseContinuous

        # to initialize/register the stacked module buffer.
        self.cache_masks()

    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self._subnet_class.apply(self.scores[j].abs(), self.sparsity)
                    for j in range(pargs.num_tasks)
                ]
            ),
        )

    def clear_masks(self):
        self.register_buffer("stacked", None)

    def forward(self, x):
        if self.task < 0:
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)
        else:
            #subnet = module_util.GetSubnet.apply(
            #    self.scores[self.task].abs(), self.sparsity
            #)
            subnet = self._forward_mask()
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def _forward_mask_normal(self):
        return self._subnet_class.apply(self.scores[self.task].abs(), self.sparsity)

    def _forward_mask_linear_comb(self):
        _subnet = self.scores[self.task]
        # NOTE comment/uncomment the code block below to disble/enable the use of consolidated masks
        # in PPO_agent (in trask_train_end(...)) also comment/uncomment `consolidate_mask` function.
        if self.task < self.num_tasks_learned:
            # this is a task that has been seen before (with established/trained mask).
            # fetch mask and use (either for eval or to continue training).
            return self._subnet_class.apply(_subnet.abs(), self.sparsity)

        # otherwise, this is a new task. check if the first task
        if self.task == 0:
            # this is the first task to train. no previous task mask to linearly combine.
            return self._subnet_class.apply(_subnet.abs(), self.sparsity)

        # otherwise, a new task and it is not the first task. combine task mask with
        # masks from previous tasks.
        # note: should not update scores/masks from previous tasks. only update their coeffs/betas
        _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        assert len(_subnets) > 0, 'an error occured'
        _betas = self.betas[self.task, 0:self.task+1]
        _betas = torch.softmax(_betas, dim=-1)
        _subnets.append(_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        return self._subnet_class.apply(_subnet_linear_comb.abs(), self.sparsity)

    @torch.no_grad()
    def consolidate_mask(self):
        self.scores[self.task].data = self._consolidate_mask()
        return

    @torch.no_grad()
    def _consolidate_mask(self):
        # catch scenarios where consolidation of mask is NOT needed.
        if self.new_mask_type == NEW_MASK_RANDOM: return
        if self.task <= 0: return
        if self.task < self.num_tasks_learned:
            # re-visiting a task that has been previously learnt (no need to consolidate)
            # which should not get here though, because this secanrio should have been caught
            # task_train_end(...) method in supermask_policy.py class.
            # assert False, 'sanity check'
            return

        _subnet = self.scores[self.task]
        _subnets = [self.scores[idx].detach() for idx in range(self.task)]
        assert len(_subnets) > 0, 'an error occured'
        _betas = self.betas[self.task, 0:self.task+1]
        _betas = torch.softmax(_betas, dim=-1)
        _subnets.append(_subnet)
        assert len(_betas) == len(_subnets), 'an error ocurred'
        _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
        # element wise sum of various masks (weighted sum)
        _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
        return _subnet_linear_comb.data

    def __repr__(self):
        return f"MultitaskMaskConv2dSparse({self.in_channels}, {self.out_channels})"

    @torch.no_grad()
    def get_mask(self, task, raw_score=True):
        # return raw scores and not the processed mask, since the
        # scores are the parameters that will be trained in other
        # agents. the binary masks would not be trained but rather
        # generated from raw scores in other agents
        if raw_score:
            # Check if we are using MASK RI or MASK LC
            if self.new_mask_type == NEW_MASK_RANDOM:
                return self.scores[task]

            # MASK LC
            else:
                # If mask requested is being trained on then
                if task < self.task:
                    return self.scores[task]

                elif task == self.task:
                    # Assuming two agents are combining 
                    return self.consolidate_mask()

                else:
                    # Requesting a mask where the agent hasn't trained for the task
                    raise ValueError('sanity check')
                    # return self.scores[task]  # Return random initialised mask
        else:
            return self._subnet_class.apply(self.scores[task])

    @torch.no_grad()
    def set_mask(self, mask, task):
        self.scores[task].data = mask
        # NOTE, this operation might not be required and could be remove to save compute time
        self.cache_masks()
        return

    @torch.no_grad()
    def set_task(self, task, new_task=False):
        self.task = task
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task:
            if task > 0:
                k = task + 1
                self.betas.data[task, 0:k] = 1. / k

