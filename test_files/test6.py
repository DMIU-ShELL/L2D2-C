import torch
from torch import nn
import time
import math

cm = [3, 3, 3]
#cm = [1, 2, 4, 3, 2, 5, 0]
K = max(cm)
T = len(cm)
betas = nn.Parameter(torch.zeros(T, T+K).type(torch.float32))


for t, k in enumerate(cm):
    betas.data[t, 0:t+1] = 1/(2*t+2)

print(betas)

for t, k in enumerate(cm):
    #betas.data[t, 0:t] = 1 / (3*max(1,t))  # Set weights of LL masks and subnet with equal probability = 0.5 (or =1 if there are no community masks)
    betas.data[t, t:t+1] = 1/3
    betas.data[t, t+1:t+1+k] = 1 / (3*max(1,k))  # Set weights of community masks to 0.5 / k (if k = 0 then nothing will happen)
    betas.data[t, 0:t+1+k] = torch.log(betas.data[t, 0:t+1+k])


    # Select weights for task
    _betas = betas.data[t, 0:t+1+k]
    print(_betas)
    _betas = torch.softmax(_betas, dim=-1)
    print(_betas)
    print()

