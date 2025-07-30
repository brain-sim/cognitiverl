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


class MLPTD3Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        a_max: float,
        a_min: float,
        hidden_dims: list[int] = [512, 256, 128],
        device: torch.device | None = None,
        activation: type[nn.Module] = nn.ELU,
        output_activation: type[nn.Module] | None = nn.Tanh,
        exploration_noise: float = 1.0,
    ):
        super().__init__()

        # Ensure device is properly handled for CudaGraph
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        actor_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                actor_layers.append(nn.Linear(n_obs, hidden_dims[i]))
            else:
                actor_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            actor_layers.append(activation())
        actor_layers.append(nn.Linear(hidden_dims[-1], n_act))
        if output_activation is not None:
            actor_layers.append(output_activation())
        self.actor = nn.Sequential(*actor_layers)
        self.actor.to(device)

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (a_max - a_min) / 2.0,  # type: ignore
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (a_max + a_min) / 2.0,  # type: ignore
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
        )
        self.register_buffer(
            "exploration_noise",
            torch.tensor(exploration_noise, dtype=torch.float32, device=device),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs) * self.action_scale + self.action_bias

    def explore(self, obs: torch.Tensor) -> torch.Tensor:
        act = self(obs)
        return act + torch.randn_like(act).mul(
            self.action_scale * self.exploration_noise
        )


class MLPTD3Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dims: list[int] = [512, 256, 128],
        device: torch.device | None = None,
    ):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        critic_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                critic_layers.append(nn.Linear(n_obs + n_act, hidden_dims[i]))
            else:
                critic_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            critic_layers.append(nn.ELU())
        critic_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
        self.critic.to(device)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], 1)
        return self.critic(x)


class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        horizon: int = 1,  # Add horizon parameter
        hidden_dims: list[int] = [1024, 512, 256],
        activation: type[nn.Module] = nn.ReLU,
        device: torch.device = None,
    ):
        super().__init__()
        self.horizon = horizon
        qnet_layers = []
        # Input is obs + (horizon * n_act) actions
        input_dim = n_obs + (horizon * n_act)

        for i in range(len(hidden_dims)):
            if i == 0:
                qnet_layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                qnet_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            qnet_layers.append(activation())
        qnet_layers.append(nn.Linear(hidden_dims[-1], num_atoms))
        self.qnet = nn.Sequential(*qnet_layers)
        self.qnet.to(device)
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # If horizon > 1, actions should be (batch_size, horizon * n_act)
        # If horizon == 1, actions should be (batch_size, n_act) - backward compatibility
        if self.horizon > 1 and actions.dim() == 3:
            # actions is (batch_size, horizon, n_act), flatten it
            batch_size = actions.shape[0]
            actions = actions.view(batch_size, -1)
        x = torch.cat([obs, actions], 1)
        x = self.qnet(x)
        return x

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        l_mask = torch.logical_and((u > 0), (l == u))
        u_mask = torch.logical_and((l < (self.num_atoms - 1)), (l == u))

        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)
        next_dist = F.softmax(
            self.forward(obs, actions), dim=1
        )  # TODO: Check if softmax is calculated on correct dimension
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )  # TODO: Check if offset is calculated correctly if batch_size is 1
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
        if proj_dist.isnan().any() or proj_dist.isinf().any():
            raise ValueError("proj_dist nan or inf")
        return proj_dist


class MLPFastTD3Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        horizon: int = 1,  # Add horizon parameter
        hidden_dims: list[int] = [512, 256, 128],
        activation: type[nn.Module] = nn.ReLU,
        device: torch.device = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.qnet1 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            horizon=horizon,  # Pass horizon parameter
            hidden_dims=hidden_dims,
            activation=activation,
            device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            horizon=horizon,  # Pass horizon parameter
            hidden_dims=hidden_dims,
            activation=activation,
            device=device,
        )

        self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
        )
        self.device = device

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        q1_proj = self.qnet1.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        q2_proj = self.qnet2.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        return q1_proj, q2_proj

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        sum_probs = torch.sum(
            probs * self.q_support, dim=1
        )  # TODO: Check if sum is calculated on correct dimension
        return sum_probs


class MLPFastTD3Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        horizon: int = 1,  # Add horizon parameter
        hidden_dims: list[int] = [512, 256, 128],
        activation: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] | None = nn.Tanh,
        std_min: float = 0.05,
        std_max: float = 0.8,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.n_act = n_act
        self.horizon = horizon

        actor_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                actor_layers.append(nn.Linear(n_obs, hidden_dims[i]))
            else:
                actor_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            actor_layers.append(activation())
        self.net = nn.Sequential(*actor_layers)
        self.net.to(device)

        # Output horizon * n_act actions when horizon > 1, otherwise n_act
        output_dim = horizon * n_act if horizon > 1 else n_act
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            output_activation() if output_activation is not None else nn.Identity(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        self.fc_mu.to(device)

        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)

        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        When horizon > 1: Output action sequence π(s) -> a_{t:t+h}
        Returns: (batch_size, horizon, n_act) if horizon > 1, else (batch_size, n_act)
        """
        x = self.net(x)
        x = self.fc_mu(x)

        if self.horizon > 1:
            batch_size = x.shape[0]
            return x.view(batch_size, self.horizon, self.n_act)
        else:
            return x

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )

        act = self(obs)
        if deterministic:
            return act

        if self.horizon > 1:
            # Broadcast noise over horizon dimension
            noise = torch.randn_like(act) * self.noise_scales.unsqueeze(1)
        else:
            noise = torch.randn_like(act) * self.noise_scales
        return act + noise
