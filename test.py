import torch

mytensor = torch.ones(10,3) * torch.inf

#print(mytensor)
#print(type(mytensor))


myothertensor = torch.tensor([0., 0., 1.])

mytensor[0] = myothertensor
print(mytensor)