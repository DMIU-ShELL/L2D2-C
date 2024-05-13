import torch
import torch.nn.functional as F

mask1 = torch.tensor([0.6554, 0.2804, 0.5919, 0.0928, 0.6591])
mask2 = torch.tensor([0.9135, 0.6304, 0.8435, 0.0876, 0.3023])
mask3 = torch.tensor([0.1260, 0.5552, 0.5050, 0.0033, 0.6471])

_subnet = torch.tensor([0.1420, 0.4843, 0.1653, 0.5874, 0.2608])
_subnets = [mask1, mask2, mask3]

_subnets.append(_subnet)
print('_subnets: ', _subnets, '\n')


_betas = torch.zeros(len(_subnets))
########################################################################
# mask BLC modification
_betas[-1] = 0.5
k = len(_subnets) - 1   # num masks excluding the current training mask
_betas[:-1] = 1 / (2 * k)
########################################################################
_betas = torch.softmax(_betas, dim=-1)


_subnets = [_b * _s for _b, _s in zip(_betas, _subnets)]
print('betas applied: ', _subnets, '\n')

_subnet_linear_comb_sum = torch.stack(_subnets, dim=0).sum(dim=0)
_subnet_linear_comb_mean = torch.stack(_subnets, dim=0).mean(dim=0)

print('sum: ', _subnet_linear_comb_sum)
print('mean: ', _subnet_linear_comb_mean)
print('og mask:', _subnet, '\n')

"""# Compute Euclidean distance
euclidean_distance_sum = torch.norm(_subnet_linear_comb_sum - _subnet)
euclidean_distance_mean = torch.norm(_subnet_linear_comb_mean - _subnet)

# Compute cosine similarity
cosine_similarity_sum = F.cosine_similarity(_subnet_linear_comb_sum.unsqueeze(0), _subnet.unsqueeze(0))
cosine_similarity_mean = F.cosine_similarity(_subnet_linear_comb_mean.unsqueeze(0), _subnet.unsqueeze(0))

print('Euclidean Distance (Sum): ', euclidean_distance_sum)
print('Euclidean Distance (Mean): ', euclidean_distance_mean)
print('Cosine Similarity (Sum): ', cosine_similarity_sum.item())
print('Cosine Similarity (Mean): ', cosine_similarity_mean.item())"""
