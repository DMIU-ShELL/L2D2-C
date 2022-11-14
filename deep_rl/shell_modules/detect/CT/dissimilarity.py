import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import ot
from plotting import *

class dissimilarity:
  def __init__(self, title, reference, device, num_samples=16384, input_dim=144, num_iter=10, one_hot=True, normalized=True, demo=True):
    assert reference is not None, f'Reference not found.'
    self.ref = reference
    self.device = device
    self.num_samples = num_samples
    self.num_iter = num_iter
    self.oh = one_hot
    self.normalized = normalized
    self.demo = demo
    self.title = title
    self.input_dim = input_dim

  def preprocess_dataset(self, X):
    if self.num_samples is not None and len(X) > self.num_samples:
      idxs = np.sort(np.random.choice(len(X), self.num_samples, replace=False))
      sampler = SubsetRandomSampler(idxs)
      loader = DataLoader(X, sampler=sampler, batch_size=64)
    else:
      ## No subsampling
      loader = DataLoader(X, batch_size=64)
        
    X = []

    for batch in loader:
      X.append(batch.squeeze().view(batch.shape[0],-1))
    X = torch.cat(X).to(self.device)

    img = X[:,:self.input_dim]
    act = X[:,self.input_dim:-1]
    reward = X[:,-1].unsqueeze(1)

    if not self.normalized:
      mean = torch.mean(img.float())
      std = torch.std(img.float())
      img = (img.float()-mean)/std
    # if not self.oh:
    #   act_oh = torch.zeros((X.shape[0],len(torch.unique(act))))
    #   for i in range(act.shape[0]):
    #     act_oh [i,int(act[i])]=1
    #   act = act_oh.to(self.device)
    #   # lb = preprocessing.LabelBinarizer()
    #   # lb.fit(act_.cpu())
    #   # act = lb.transform(act_.cpu())
    return torch.cat((img, act, reward), dim=1).float()
    # return X.float()

  def lwe(self, X):
    X = self.preprocess_dataset(X)
    ref_size = self.ref.shape[0]
    C = ot.dist(X.cpu(), self.ref).cpu().numpy()
    # Calculating the transport plan
    gamma = torch.from_numpy(ot.emd(ot.unif(X.shape[0]), ot.unif(ref_size), C, numItermax=700000)).float()
    # Calculating the transport map via barycenter projection /gamma.sum(dim=0).unsqueeze(1)
    f=(torch.matmul((ref_size*gamma).T,X.cpu())-self.ref)/np.sqrt(ref_size)
    return f

  def pwdist(self, tasks_dict):
    num_tasks = len(tasks_dict)
    tasks = tasks_dict.values()
    task_ids = list(tasks_dict.keys())

    #initialize pairwise distance matrix
    dist = torch.zeros((num_tasks,num_tasks))

    for k in tqdm(range(int(self.num_iter/2))):
      task_vecs = []
      task_vecs_ = []
      for task in tasks:
        vec = self.lwe(task)
        task_vecs.append(vec)
      for i in range(num_tasks):
        for j in range(i+1,num_tasks):
            dist[i,j]+=torch.linalg.vector_norm(task_vecs[i]-task_vecs[j])/2.
      for task in tasks:
        vec = self.lwe(task)
        task_vecs_.append(vec)
      for i in range(num_tasks):
        dist[i,i]+=torch.linalg.vector_norm(task_vecs[i]-task_vecs_[i])
      for i in range(num_tasks):
        for j in range(i+1,num_tasks):
            dist[i,j]+=torch.linalg.vector_norm(task_vecs_[i]-task_vecs_[j])/2.

    #Filling in the lower triangle of the matrix
    for i in range(1, num_tasks):
      for j in range(i):
        dist[i,j] = dist[j,i]

    if self.demo:
      fig, ax = plt.subplots(figsize=(num_tasks+2, num_tasks+2))

      im, cbar = heatmap(self.title, dist, task_ids, task_ids, ax=ax,
                    cmap="RdPu")
      texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=18)

      fig.tight_layout()
      plt.show()
      fig.savefig(self.title, format='png', dpi=1000)

    return dist