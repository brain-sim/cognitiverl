from typing import List

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class CNNPPOAgent(nn.Module):
    """
    Convolutional agent using a pretrained MobileNetV3 backbone for image feature
    extraction, followed by fully connected layers for policy and value estimation.
    """

    def __init__(
        self,
        n_obs,
        n_act,
        img_size: List[int] = [3, 32, 32],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        noise_std_type="scalar",
        init_noise_std=1.0,
    ):
        super().__init__()
        self.noise_std_type = noise_std_type
        # Image input dimensions
        channels, height, width = img_size
        self.img_size = (channels, height, width)

        # Load and adapt MobileNetV3-small backbone
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone.eval()  # freeze backbone in eval mode

        # Adjust first conv layer for 32x32 inputs
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=channels,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.backbone = nn.Sequential(*list(self.backbone.features))

        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            feature_size = self.backbone(dummy).view(1, -1).size(1)

        # MLP for extracted features
        actor_layers = []
        critic_layers = []
        for i in range(len(actor_hidden_dims)):
            if i == 0:
                actor_layers.append(nn.Linear(feature_size, actor_hidden_dims[i]))
                critic_layers.append(nn.Linear(feature_size, critic_hidden_dims[i]))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[i - 1], actor_hidden_dims[i])
                )
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[i - 1], critic_hidden_dims[i])
                )
            actor_layers.append(nn.ELU())
            critic_layers.append(nn.ELU())
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], n_act))
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)

        if self.noise_std_type == "scalar":
            self.actor_std = nn.Parameter(init_noise_std * torch.ones(n_act))
        elif self.noise_std_type == "log":
            self.actor_std = nn.Parameter(torch.log(init_noise_std * torch.ones(n_act)))
        else:
            raise ValueError(f"Invalid noise_std_type: {self.noise_std_type}")
        Normal.set_default_validate_args(False)

    def extract_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from image portion of the state vector."""
        bsz = x.size(0)
        c, h, w = self.img_size
        # reshape and forward through backbone
        imgs = x[:, : c * h * w].view(bsz, c, h, w)
        with torch.no_grad():
            feats = self.backbone(imgs)
        return feats.view(bsz, -1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-value from raw input."""
        img_feats = self.extract_image(x)
        return self.critic(img_feats)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action from raw input."""
        img_feats = self.extract_image(x)
        return self.actor(img_feats)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple:
        """Compute action, log-prob, entropy, and value for input states."""
        img_feats = self.extract_image(x)
        action_mean = self.actor(img_feats)
        action_std = self.actor_std.expand_as(action_mean)
        if self.noise_std_type == "log":
            action_std = torch.clamp(action_std, -20.0, 5.0)
            action_std = torch.exp(action_std)
        elif self.noise_std_type == "scalar":
            action_std = torch.clamp(action_std, min=1e-6)
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.critic(img_feats),
            action_mean,
            action_std,
        )

    def forward(self, obs):
        return self.get_action(obs)
