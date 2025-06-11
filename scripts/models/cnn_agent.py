import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .utils import layer_init


class CNNPPOAgent(nn.Module):
    """
    Convolutional agent using a pretrained MobileNetV3 backbone for image feature
    extraction, followed by fully connected layers for policy and value estimation.
    """

    def __init__(self, n_obs, n_act):
        super().__init__()

        # Image input dimensions
        channels, height, width = 3, 32, 32
        self.img_size = (channels, height, width)

        # Load and adapt MobileNetV3-small backbone
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        backbone.eval()  # freeze backbone in eval mode

        # Adjust first conv layer for 32x32 inputs
        backbone.features[0][0] = nn.Conv2d(
            in_channels=channels,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.backbone = nn.Sequential(*backbone.features)

        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            feat_dim = self.backbone(dummy).view(1, -1).size(1)

        # MLP for extracted features
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(feat_dim, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 64)),
            nn.ELU(),
        )

        # Critic head
        self.critic = layer_init(nn.Linear(64, 1), std=0.0, bias_const=0.0)

        # Actor head (mean) and log std parameter
        self.actor_mean = layer_init(nn.Linear(64, n_act), std=1.0)
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act))

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
        h = self.feature_net(img_feats)
        return self.critic(h)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> tuple:
        """Compute action, log-prob, entropy, and value for input states."""
        img_feats = self.extract_image(x)
        h = self.feature_net(img_feats)

        mean = self.actor_mean(h)
        logstd = self.actor_logstd.expand_as(mean)
        logstd = torch.clamp(logstd, -20, 2)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        if action is None:
            action = dist.rsample()

        logprob = dist.log_prob(action).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        value = self.critic(h).view(-1)

        return action, logprob, entropy, value
