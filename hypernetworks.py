
from utils import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def sample_dirichlet(K):
    dist = torch.distributions.Dirichlet(torch.ones(K))
    return dist.sample()

class BaseNetwork(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__() # BaseNetwork, self
        self.g = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # Maps R^K -> R^d
        )

    def forward(self, z):
        return self.g(z)  # Output of g(z)

class PredNetwork(nn.Module):
    def forward(self, x, g_z):
        return torch.matmul(x, g_z.T)  # x.T @ g(z)
    
class HyperNetwork(nn.Module):
    def __init__(self, K, d):
        super().__init__() # HyperNetwork, self
        self.K = K
        self.d = d
        self.basenetwork = BaseNetwork(K, d)
        self.prednetwork = PredNetwork()
        self.evals = None
    
    def forward(self,x,z):
        return self.prednetwork(x,self.basenetwork(z))
    
    def eval_square_loss(self,theta1,theta2,cov1,cov2, lambdas):
        losses = np.empty(shape=(len(lambdas),2))
        for i in range(len(lambdas)):
            lamb = torch.tensor([lambdas[i],1-lambdas[i]],dtype =torch.float)
            params = self.basenetwork(lamb).detach()
            losses[i,0] = pop_square_loss(params,theta1,cov1)
            losses[i,1] = pop_square_loss(params,theta2,cov2)
        self.evals = losses

def create_hypernetwork_dataset(X1, X2, y1, y2):
    dataset = {
            "x_task_1": torch.tensor(X1,dtype = torch.float),
            "x_task_2": torch.tensor(X2,dtype =torch.float),
            "y_task_1": torch.tensor(y1,dtype =torch.float),
            "y_task_2": torch.tensor(y2,dtype =torch.float)
        }
    return dataset

def create_hypernetwork_dataset_pseudo(X1_labeled, X2_labeled, X1_unlabeled,X2_unlabeled, y1, y2, y1_pseudo, y2_pseudo):
    dataset = {
        "x_task_1": torch.concatenate((torch.tensor(X1_labeled,dtype = torch.float),torch.tensor(X1_unlabeled,dtype = torch.float)),axis=0),
        "x_task_2": torch.concatenate((torch.tensor(X2_labeled,dtype = torch.float),torch.tensor(X2_unlabeled,dtype = torch.float)),axis=0),
        "y_task_1": torch.concatenate((torch.tensor(y1,dtype =torch.float),torch.tensor(y1_pseudo,dtype =torch.float)),axis=0),
        "y_task_2": torch.concatenate((torch.tensor(y2,dtype =torch.float),torch.tensor(y2_pseudo,dtype =torch.float)),axis=0)
    }
    return dataset

def train(hypernetwork, num_epochs, dataset, reg_strength_fun, verbose=False):
    optimizer = torch.optim.Adam(hypernetwork.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        # Sample latent variables for both tasks
        lamb = sample_dirichlet(hypernetwork.K)  # Random latent variables

        # Predictions for both tasks
        predictions_task_1 = hypernetwork(dataset["x_task_1"], lamb)
        predictions_task_2 = hypernetwork(dataset["x_task_2"], lamb)

        # Compute losses
        loss_task_1 = loss_fn(predictions_task_1, dataset["y_task_1"])
        loss_task_2 = loss_fn(predictions_task_2, dataset["y_task_2"])

        # Total loss
        total_loss = lamb[0]*loss_task_1 + lamb[1]*loss_task_2 + reg_strength_fun(lamb[0]) * torch.norm(hypernetwork.basenetwork(lamb), p=1)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log progress
        if verbose:
            print(f"Epoch {epoch + 1}, Loss: {total_loss.item():.4f}")

if __name__ == '__main__':
    raise RuntimeError('no main')
