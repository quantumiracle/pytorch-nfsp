import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math

from functools import partial

def DQN(env, args):
    if args.dueling:
        model = DuelingDQN(env, args.hidden_dim)
    else:
        model = DQNBase(env, args.hidden_dim)
    return model

class DQNBase(nn.Module):
    """
    Basic DQN

    parameters
    ---------
    env         environment(openai gym)
    """
    def __init__(self, env, hidden_dim=64):
        super(DQNBase, self).__init__()
        
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.flatten = Flatten()
        self.hidden_dim = hidden_dim
        if len(self.input_shape) > 1: # image
            self.features = nn.Sequential(
                nn.Conv2d(self.input_shape[0], 8, kernel_size=4, stride=2),
                cReLU(),
                nn.Conv2d(16, 8, kernel_size=5, stride=1),
                cReLU(),
                nn.Conv2d(16, 8, kernel_size=3, stride=1),
                cReLU()
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(self.input_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, int(hidden_dim/2)),
                nn.ReLU(),
            )
        
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), self.num_actions)
        )
        
    def forward(self, x):
        x /= 255.
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def _feature_size(self):
        # print(torch.zeros(1, *self.input_shape).shape)
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_value = self.forward(state)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action

class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env, hidden_dim=64):
        super(DuelingDQN, self).__init__(env, hidden_dim)
        
        self.advantage = self.fc

        self.value = nn.Sequential(
            nn.Linear(self._feature_size(), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), 1)
        )
    
    def forward(self, x):
        x /= 255.
        x = self.features(x)
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

class Policy(DQNBase):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """
    def __init__(self, env, hidden_dim=64):
        super(Policy, self).__init__(env, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), self.num_actions),
            nn.Softmax(dim=1)
        )

    def act(self, state):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)
            action = distribution.multinomial(1).item()
        return action


class cReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

