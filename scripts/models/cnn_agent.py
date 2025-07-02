from typing import List

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .base_agent import BaseAgent


class CNNPPOAgent(BaseAgent):
    """
    CNN PPO Agent using MobileNetV3 backbone.
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        img_size: List[int] = [3, 32, 32],
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

        # Image dimensions
        channels, height, width = img_size
        self.img_size = (channels, height, width)

        # Setup backbone and get feature size
        self._setup_backbone(channels)

        # Build networks using base class method
        self.actor = self.build_networks(
            input_dim=self.feature_size,
            output_dim=n_act,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        self.critic = self.build_networks(
            input_dim=self.feature_size,
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

    def _setup_backbone(self, channels: int):
        """Setup MobileNetV3 backbone."""
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone.eval()

        # Adjust first conv layer
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=channels,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.backbone = nn.Sequential(*list(self.backbone.features))

        # Get feature size
        with torch.no_grad():
            dummy = torch.zeros(1, *self.img_size)
            self.feature_size = self.backbone(dummy).view(1, -1).size(1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from image input."""
        batch_size = x.size(0)
        c, h, w = self.img_size

        # Reshape to image format
        imgs = x[:, : c * h * w].view(batch_size, c, h, w)

        # Extract features
        with torch.no_grad():
            features = self.backbone(imgs)

        return features.view(batch_size, -1)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action from input."""
        features = self.extract_features(x)
        return self.actor(features)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-value from input."""
        features = self.extract_features(x)
        return self.critic(features)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple:
        """Compute action, log-prob, entropy, and value."""
        features = self.extract_features(x)
        action_mean = self.actor(features)
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
            self.critic(features),
            action_mean,
            action_std,
        )

    def forward(self, x):
        return self.get_action(x)

    def load_from_checkpoint(self, checkpoint_path: str, load_ema: bool = False):
        """
        Load model weights from a checkpoint, handling layers that change with image size.

        Args:
            checkpoint_path (str): Path to the checkpoint file
            load_ema (bool): Whether to load EMA weights instead of regular weights
        """
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if load_ema and "ema_model_state_dict" in checkpoint:
            checkpoint_state_dict = checkpoint["ema_model_state_dict"]
            print("Loading EMA weights from checkpoint")
        elif "model_state_dict" in checkpoint:
            checkpoint_state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint_state_dict = checkpoint["state_dict"]
        else:
            checkpoint_state_dict = checkpoint

        # Get current model state dict
        current_state_dict = self.state_dict()

        # Define layers to skip (these depend on image size)
        skip_keys = {
            "backbone.0.0.weight",  # First conv layer
            "actor.0.weight",  # First actor layer
            "actor.0.bias",  # First actor layer bias
            "critic.0.weight",  # First critic layer
            "critic.0.bias",  # First critic layer bias
        }

        # Create new state dict, skipping size-dependent layers
        new_state_dict = {}
        skipped_layers = []

        for key, value in checkpoint_state_dict.items():
            # Skip layers that depend on image size
            if key in skip_keys:
                skipped_layers.append(key)
                print(f"Skipping size-dependent layer: {key}")
                continue

            # All other layers must match exactly
            if key not in current_state_dict:
                raise KeyError(f"Key {key} from checkpoint not found in current model")

            if current_state_dict[key].shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"checkpoint {value.shape} vs model {current_state_dict[key].shape}"
                )

            new_state_dict[key] = value

        # Verify all current model keys (except skipped ones) are present in checkpoint
        for key in current_state_dict.keys():
            if key in skip_keys:
                continue
            if key not in checkpoint_state_dict:
                raise KeyError(f"Key {key} from current model not found in checkpoint")

        # Load the weights (strict=False because we're skipping some layers)
        self.load_state_dict(new_state_dict, strict=False)

        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        print(f"Loaded {len(new_state_dict)} layers")
        print(f"Skipped {len(skipped_layers)} size-dependent layers: {skipped_layers}")

        return True
