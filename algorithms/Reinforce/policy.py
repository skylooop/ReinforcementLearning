import torch
import torch.nn as nn
from absl import flags
import os.path, sys
import typing as tp
import numpy as np

from torch.distributions import Normal

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from basicAgent import BasicAgent



FLAGS = flags.FLAGS


class PolicyNet(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        '''
        Predicting mean and variance parameters to then sample actions from Normal distribution
        '''
        
        super().__init__()
        
        self.hidden_size1 = FLAGS.hidden_size_first
        self.hidden_size2 = FLAGS.hidden_size_second
        
        self.model = nn.Sequential(
            nn.Linear(obs_size, self.hidden_size1),
            nn.Tanh(),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.Tanh()
        )
        
        self.policy_mean_net = nn.Sequential(
            nn.Linear(self.hidden_size2, act_size)
        )
        
        self.variance_net = nn.Sequential(
            nn.Linear(self.hidden_size2, act_size)
        )       
        
    def forward(self, state: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.model(state.float())
        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.variance_net(shared_features))
        )
        
        return action_means, action_stddevs


class Reinforce(BasicAgent):
    def __init__(self, obs_size: int, act_size: int):
        
        self.gamma = FLAGS.gamma
        self.lr = FLAGS.lr
        self.eps = FLAGS.eps
        
        self.probs = []
        self.rewards = []
        
        self.net = PolicyNet(obs_size, act_size).cuda()
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        
    def sample_action(self, state: np.ndarray) -> float:
        state = torch.tensor(np.array([state])).cuda()
        action_means, action_stddevs = self.net(state)

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)
        
        action = action.cpu().detach().numpy()
        self.probs.append(prob)

        return action
    
    def update(self):
        '''
        Update policy
        '''
        running_g = 0
        gs = []
        
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)
        loss = 0
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)
            
         # Update the policy network
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.probs = []
        self.rewards = []