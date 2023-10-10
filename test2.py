import torch

T = []
for i in range(3, 10):
    tensor_size = 5  # Adjust the size of the tensors as needed
    random_tensor = torch.FloatTensor(1, tensor_size).uniform_(-10, 10)  # Generate a random tensor
    T.append(random_tensor)

DIST_THRESHOLD = 12

t_avg = T[0]
t_dist = 0

for new in T:
    t_avg = t_avg + new / 2
    t_dist_prev = t_dist
    t_dist = torch.linalg.norm(t_avg - new)
    t_sim_avg = t_dist_prev + t_dist / 2




    if t_sim_avg > DIST_THRESHOLD:
        print('\nNEW TASK DETECTED')
        t_dist = 0
        t_sim_avg = 0
        
        t_avg = t_avg + new / 2
        t_dist_prev = t_dist
        t_dist = torch.linalg.norm(t_avg - new)
        t_sim_avg = t_dist_prev + t_dist / 2

    print(f'Tensor: {new}\nTensor avg: {t_avg}\nSimilarity: {t_dist}\nSimilarity avg: {t_sim_avg}\n')

