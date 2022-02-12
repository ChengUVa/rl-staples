import numpy as np
import torch
import torch.nn as nn
from utils import float32_preprocessor
HID_SIZE = 200

class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size, val_scale=1.0):
        super().__init__()
        self.val_scale = val_scale

        self.base = nn.Sequential(nn.Linear(obs_size, HID_SIZE), nn.ReLU(),)

        # policy head
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size), nn.Tanh(),  # in range (-1,1)
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size), nn.Softplus(),  # smoothed RELU, positive
        )

        # value function head
        # self.value = nn.Linear(HID_SIZE, 1)  # no activation
        self.value = nn.Sequential(
            nn.Linear(HID_SIZE, HID_SIZE//2),
            nn.ReLU(),
            nn.Linear(HID_SIZE//2, 1)
        )

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out) * self.val_scale


class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(), 
            #nn.Linear(HID_SIZE, HID_SIZE),
            #nn.ReLU(), 
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(), # in range (-1, 1)
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)


class ModelCritic(nn.Module):
    def __init__(self, obs_size, val_scale=1.0):
        super(ModelCritic, self).__init__()
        self.val_scale = val_scale

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            #nn.Linear(HID_SIZE, HID_SIZE),
            #nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x) * self.val_scale


class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


class AgentA2C(BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        """convert observation to actins"""
        states_v = float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)  # sample action from normal dist.
        actions = np.clip(actions, -1, 1)  # action value between -1 and 1
        # agent_states is not used for sampling actions here
        return actions, agent_states