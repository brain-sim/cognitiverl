import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .utils import layer_init


class MLPAgent(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, n_act), std=1.0),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_logstd = torch.clamp(action_logstd, -20, 2)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(obs),
        )
