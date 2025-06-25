import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

class MLPPPOAgent(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation=nn.ELU,
        noise_std_type="scalar",
        init_noise_std=1.0,
    ):
        super().__init__()
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.actor_std = nn.Parameter(init_noise_std * torch.ones(n_act))
        elif noise_std_type == "log":
            self.actor_std = nn.Parameter(torch.log(init_noise_std * torch.ones(n_act)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")
        critic_layers = []
        actor_layers = []
        for i in range(len(actor_hidden_dims)):
            if i == 0:
                actor_layers.append(nn.Linear(n_obs, actor_hidden_dims[i]))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[i - 1], actor_hidden_dims[i])
                )
            actor_layers.append(activation())
        for i in range(len(critic_hidden_dims)):
            if i == 0:
                critic_layers.append(nn.Linear(n_obs, critic_hidden_dims[i]))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[i - 1], critic_hidden_dims[i])
                )
            critic_layers.append(activation())
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], n_act))
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)
        Normal.set_default_validate_args(False)

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action from raw input."""
        return self.actor(x)

    def get_action_and_value(self, obs, action : torch.Tensor | None =None):
        action_mean = self.actor(obs)
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
            self.critic(obs),
            action_mean,
            action_std,
        )
    
    def forward(self, obs):
        return self.get_action(obs)
