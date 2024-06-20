import numpy as np
import ot
import torch
#from plotting import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import random

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

        self.embeddings = []
        self.rewards = []


        n_components = 1
        self.IncrementalPCA = IncrementalPCA(n_components=n_components)

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

    def preprocess_dataset(self, X, some_task_action_space_size):
        '''Function that preprpcess the Data-Batch of SAR before calcuting the embedding'''
        if self.num_samples is not None and len(X) > self.num_samples:
            idxs = np.sort(np.random.choice(len(X), self.num_samples, replace=False))
            sampler = SubsetRandomSampler(idxs)
            loader = DataLoader(X, sampler=sampler, batch_size=64)
        else:
            ## No subsampling
            loader = DataLoader(X, batch_size=64)
            
        X = []
        q=0
        for batch in loader:
            q = q+1
            X.append(batch.squeeze().view(batch.shape[0],-1))
        X = torch.cat(X).to(self.device)
        #print("QQQQQQQQQQQQ:", q)
        #print("FIRST X:", X)

        img = X[:,:self.input_dim]
        act = X[:,self.input_dim:-1]
        reward = X[:,-1].unsqueeze(1)

        if self.normalized:
            mean = torch.mean(img.float())
            std = torch.std(img.float())
            img = (img.float()-mean)/std
        if self.oh:
            act_oh = torch.zeros(X.shape[0], some_task_action_space_size)
            #print("INITIAL ACT_OH:", act_oh, act_oh.shape)
            #print(act, act.shape)
      
            for i in range(act.shape[0]):
                #print("HI form i:", i)
                act_oh [i,int(act[i])]=1
            act = act_oh.to(self.device)
            #print("act_OH:", act_oh)
            #lb = preprocessing.LabelBinarizer()
            #lb.fit(act_.cpu())
            #act = lb.transform(act_.cpu())
        return torch.cat((img, act, reward), dim=1).float()
        # return X.float()

    def lwe(self, X, some_task_action_space_size):
        '''Calculates the Embedding for a given Data-Batch of SAR
        Returns a 1D Tensor with the calculated Embedding'''
        X = self.preprocess_dataset(X, some_task_action_space_size)
        #print("PrePRocess_SAR_DETECT:", X)
        #print("What we actually use form the data we give:", X.shape)
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

    def mahalanobis_distance(self, current_embedding, new_embedding, cov_matrix):
        mahalanobis_dist = mahalanobis(new_embedding, current_embedding, cov_matrix)

        return mahalanobis_dist
    
    def cosine_sim(self, current_embedding, new_embedding):
        cosine_similarity = F.cosine_similarity(new_embedding, current_embedding, dim=0)
        return cosine_similarity
    
    def estimate_density(self, current_embeddings, new_embedding, bandwidth=0.5):
        # Combine current embeddings with new embedding
        all_embeddings = np.vstack([current_embeddings, new_embedding])
        
        # Fit Kernel Density Estimation model
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(all_embeddings)
        
        # Estimate density for new embedding
        density = np.exp(kde.score_samples(new_embedding.reshape(1, -1)))
        
        return density
    
    def wasserstein_distance(self, current_embeddings, new_embedding):
        # Compute Earth Mover's Distance
        emd = wasserstein_distance(current_embeddings, new_embedding)
        
        return emd


    # NOTE: WEIGHTED AVG COSINE SIM
    def add_embedding(self, embedding, reward):
        self.embeddings.append(embedding)
        self.rewards.append(reward)

    def calculate_weighted_average(self):
        if not self.embeddings or not self.rewards:
            raise ValueError("No embeddings or rewards added yet.")

        # Convert rewards to PyTorch tensor for easier manipulation
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float)

        # Add a small epsilon value to prevent division by zero
        epsilon = torch.tensor(1e-8)
        total_reward = torch.sum(rewards_tensor) + epsilon

        # Normalize rewards to sum up to 1
        normalized_rewards = rewards_tensor / total_reward

        # Calculate weighted average
        weighted_sum = torch.zeros_like(self.embeddings[0], dtype=torch.float)
        for embedding, reward in zip(self.embeddings, normalized_rewards):
            weighted_sum += reward * embedding
        weighted_average = weighted_sum / len(self.embeddings)

        return weighted_average

    def refresh_buffers(self):
       self.embeddings = []
       self.rewards = []

    def online_pca(self, embedding):
        """
        Performs online PCA dimensionality reduction on the embedding.

        Args:
            embedding: A tensor representing the embedding to be reduced.
            n_components: The number of components to retain after dimensionality reduction.

        Returns:
            A tensor representing the reduced-dimensionality embedding.
        """
        self.IncrementalPCA.partial_fit(embedding.unsqueeze(0))
        return self.IncrementalPCA.transform(embedding.unsqueeze(0)).squeeze(0)

    def random_projections(self, embedding, n_components):
        """
        Performs dimensionality reduction using random projections.

        Args:
            embedding: A tensor representing the embedding to be reduced.
            n_components: The number of components to retain after dimensionality reduction.

        Returns:
            A tensor representing the reduced-dimensionality embedding.
        """
        print(embedding.shape, n_components)
        if not hasattr(self, 'projection_matrix'):
            self.projection_matrix = torch.nn.init.xavier_normal_(torch.empty(embedding.shape[0], n_components))
        return torch.matmul(embedding, self.projection_matrix)