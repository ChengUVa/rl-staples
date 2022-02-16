import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

HID_SIZE = 200


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, HID_SIZE),
            nn.ReLU(), 
            #nn.Linear(HID_SIZE, HID_SIZE),
            #nn.ReLU(), 
            nn.Linear(HID_SIZE, act_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_dim, HID_SIZE),
            nn.ReLU(),
            #nn.Linear(HID_SIZE, HID_SIZE),
            #nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x) 


# class BaseAgent:
#     """
#     Abstract Agent interface
#     """
#     def initial_state(self):
#         """
#         Should create initial empty state for the agent. It will be called for the start of the episode
#         :return: Anything agent want to remember
#         """
#         return None

#     def __call__(self, states, agent_states):
#         """
#         Convert observations and states into actions to take
#         :param states: list of environment states to process
#         :param agent_states: list of states with the same length as observations
#         :return: tuple of actions, states
#         """
#         assert isinstance(states, list)
#         assert isinstance(agent_states, list)
#         assert len(agent_states) == len(states)

#         raise NotImplementedError