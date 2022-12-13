import numpy as np
import ot
import torch
#from plotting import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm


class Detect:
  def __init__(self, reference_num, input_dim, num_samples, reference=200,  device='cpu',title = '', num_iter=10, one_hot=True, normalized=True, demo=True):
    assert reference is not None, f'Reference not found.'
    self.ref = None
    self.device = device
    self.num_samples = num_samples
    self.num_iter = num_iter
    self.oh = one_hot
    self.normalized = normalized
    self.demo = demo
    self.title = title
    self.input_dim = input_dim
    self.reference_num = reference_num



  def set_input_dim(self, an_input_dim):
    '''A setter method, for manually setting and setting the input dimensionality of the detect
    module.'''
    self.input_dim = an_input_dim

  def get_input_dim(self):
    '''A getter method for accessing the input dimensionality of the detect module.'''
    return self.input_dim


  def set_reference(self, a_task_observation_dim, some_reference_num, some_action_dim):
    '''A setter method, for manually setting and updating the reference for calculating
    the tasks embeddings.'''
    torch.manual_seed(98)
    reference = torch.rand(some_reference_num, (a_task_observation_dim + some_action_dim + 1))#Plus one which is the reward.
    self.ref = reference

  def get_reference(self):
    '''A getter method for accessing the reference which is used to calculate the task
    embeddings'''
    return self.ref

  def set_num_samples(self, a_num_samples):
    '''A setter method for manually setting the num of samples'''
    self.num_samples = a_num_samples

  def get_num_samples(self):
    '''A getter method for retreiving the detect sample size.'''
    return self.num_samples


  def precalculate_embedding_size(self, a_reference_num, an_inputdim, some_action_dim):
    '''A method for calculating the embedding dimension '''
    pre_calc_embedding_size = a_reference_num * (an_inputdim + some_action_dim + 1)#Plus one which is the reward.
    return pre_calc_embedding_size

  def preprocess_dataset(self, X):
    '''Function that preprpcess the Data-Batch of SAR before calcuting the embedding'''
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
    print("FIRST X:", X)

    img = X[:,:self.input_dim]
    act = X[:,self.input_dim:-1]
    reward = X[:,-1].unsqueeze(1)

    if not self.normalized:
      mean = torch.mean(img.float())
      std = torch.std(img.float())
      img = (img.float()-mean)/std
    if not self.oh:
       act_oh = torch.zeros(X.shape[0], 3)
       print("INITIAL ACT_OH:", act_oh)
 
       for i in range(act.shape[0]):
        print("HI form i:", i)
        act_oh [i,int(act[i])]=1
       act = act_oh.to(self.device)
       print("act_OH:", act_oh)
       #lb = preprocessing.LabelBinarizer()
       #lb.fit(act_.cpu())
       #act = lb.transform(act_.cpu())
    return torch.cat((img, act, reward), dim=1).float()
    # return X.float()

  def lwe(self, X):
    '''Calculates the Embedding for a given Data-Batch of SAR
    Returns a 1D Tensor with the calculated Embedding'''
    X = self.preprocess_dataset(X)
    print("PrePRocess_SAR_DETECT:", X)
    print(X.shape)
    ref_size = self.ref.shape[0]
    C = ot.dist(X.cpu(), self.ref).cpu().numpy()
    # Calculating the transport plan
    gamma = torch.from_numpy(ot.emd(ot.unif(X.shape[0]), ot.unif(ref_size), C, numItermax=700000)).float()
    # Calculating the transport map via barycenter projection /gamma.sum(dim=0).unsqueeze(1)
    f=(torch.matmul((ref_size*gamma).T,X.cpu())-self.ref)/np.sqrt(ref_size)
    return f.ravel()
    
  def calculate_lwes_distance(self, lwe1, lwe2):
    '''Calculates the Euclidian Distance of the old vs the new embedding
    It returns a 2D Tensor of size (num * Data-Batch sample size)'''
    eu_dist = (lwe1 - lwe2).pow(2).ravel
    return eu_dist

  def pwdist(self, tasks_dict):
    '''Computes the Distance of the Ebeddings of Different Taks & create a similarity Matrix for all different Tasks'''
    num_tasks = len(tasks_dict)
    tasks = tasks_dict.values()
    task_ids = list(tasks_dict.keys())

    #initialize pairwise distance matrix
    dist = torch.zeros((num_tasks,num_tasks))

    for k in tqdm(range(int(self.num_iter/2))):
      task_vecs = []
      task_vecs_ = []
      #emb_list = []
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


  def emb_distance(self, current_embedding, new_calculated_embedding):
      '''Computes the Distance of the newlly calculated embedding and the one that is 
      stored for the current task the agent is solving.'''

      distance  = torch.linalg.vector_norm(current_embedding - new_calculated_embedding)

      return distance

    