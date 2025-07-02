from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .base_agent import BaseAgent


class MLPPPOAgent(BaseAgent):
    """
    MLP PPO Agent.
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        actor_hidden_dims: List[int] = [512, 256, 128],
        critic_hidden_dims: List[int] = [512, 256, 128],
        activation: type[nn.Module] = nn.ELU,
        noise_std_type: str = "scalar",
        init_noise_std: float = 1.0,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)

        self.n_obs = n_obs
        self.n_act = n_act
        self.noise_std_type = noise_std_type

        # Build networks using base class method
        self.actor = self.build_networks(
            input_dim=n_obs,
            output_dim=n_act,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        self.critic = self.build_networks(
            input_dim=n_obs,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        # Initialize noise parameters
        if noise_std_type == "scalar":
            self.actor_std = nn.Parameter(init_noise_std * torch.ones(n_act))
        elif noise_std_type == "log":
            self.actor_std = nn.Parameter(torch.log(init_noise_std * torch.ones(n_act)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")

        Normal.set_default_validate_args(False)

        # Move to device and set precision
        self.to(self.device, self.dtype)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action from input."""
        return self.actor(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-value from input."""
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple:
        """Compute action, log-prob, entropy, and value."""
        action_mean = self.actor(x)
        action_std = self.actor_std.expand_as(action_mean)

        if self.noise_std_type == "log":
            action_std = torch.clamp(action_std, -20.0, 2.0)
            action_std = torch.exp(action_std)
        elif self.noise_std_type == "scalar":
            action_std = F.softplus(action_std)

        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.critic(x),
            action_mean,
            action_std,
        )

    def forward(self, x):
        return self.get_action(x)
