"""Simple Behavioral Cloning model with MobileNet and MLP components.

This module defines a straightforward BC policy using MobileNet for image encoding
and MLPs for processing. Designed to have the same input interface as SeqFlowPolicy
but with direct action prediction instead of flow matching.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


class ImageEncoder(nn.Module):
    """MobileNet-based image encoder with ImageNet transforms."""

    def __init__(self, out_dim: int):
        super().__init__()
        # ImageNet normalization transforms (applied to 0..1 input)
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Load pretrained MobileNet V3 Small
        self.backbone = mobilenet_v3_small(pretrained=True)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the feature dimension from the classifier
        mobilenet_features = self.backbone.classifier[0].in_features

        # Replace classifier with our projection layer
        self.backbone.classifier = nn.Sequential(
            nn.Linear(mobilenet_features, 512),
            nn.LayerNorm(512),
            nn.ELU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) with values in [0, 1]
        # Apply ImageNet normalization
        x = self.transform(x)
        return self.backbone(x)


class StateMLP(nn.Module):
    """MLP for encoding state features."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ELU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceMLP(nn.Module):
    """MLP that processes flattened sequence representations."""

    def __init__(self, seq_len: int, feature_dim: int, out_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # MLP to process flattened sequence
        flattened_dim = seq_len * feature_dim
        self.net = nn.Sequential(
            nn.Linear(flattened_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.ELU(inplace=True),
            nn.Linear(out_dim * 2, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.ELU(inplace=True),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feature_dim)
        B, T, D = x.shape

        # Flatten sequence dimension
        x_flat = x.view(B, T * D)  # (B, T * feature_dim)

        # Process with MLP
        return self.net(x_flat)


class BCPolicy(nn.Module):
    """Simple Behavioral Cloning policy with MobileNet image encoding.

    Inputs:
    - state_seq: (B, T, S) or None
    - image_seq: (B, T, C, H, W) or dict of such
    - actions:   (B, A), normalized by dataset if enabled
    """

    def __init__(self, state_dim: int, action_dim: int, args):
        super().__init__()
        self.action_dim = action_dim
        self.d_model = args.d_model
        self.T = args.sequence_length
        self.dropout = nn.Dropout(args.dropout)

        # Encoders
        d_img = 512
        d_state = 128

        self.state_encoder = StateMLP(state_dim, d_state) if state_dim > 0 else None
        self.image_encoder = ImageEncoder(d_img)

        # Fusion layer
        fusion_input_dim = d_img + (d_state if state_dim > 0 else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fusion_input_dim, args.d_model),
            nn.LayerNorm(args.d_model),
            nn.ELU(inplace=True),
            nn.Linear(args.d_model, args.d_model),
        )

        # Sequence processor - processes entire sequence with MLP
        self.sequence_processor = SequenceMLP(
            seq_len=args.sequence_length * args.frame_stack,
            feature_dim=args.d_model,
            out_dim=args.d_model,
        )

        # Action head - directly predicts actions
        self.action_head = nn.Sequential(
            nn.Linear(args.d_model, args.ff_hidden),
            nn.LayerNorm(args.ff_hidden),
            nn.ELU(inplace=True),
            nn.Linear(args.ff_hidden, args.ff_hidden),
            nn.LayerNorm(args.ff_hidden),
            nn.ELU(inplace=True),
            nn.Linear(args.ff_hidden, action_dim),
            nn.Tanh(),  # Output actions in [-1, 1] range
        )

    def encode_images(self, image_seq: torch.Tensor) -> torch.Tensor:
        # image_seq: (B, T, C, H, W)
        B, T = image_seq.size(0), image_seq.size(1)
        images = image_seq.view(B * T, *image_seq.shape[2:]).contiguous(
            memory_format=torch.channels_last
        )
        feats = self.image_encoder(images)
        return feats.view(B, T, -1)

    def encode_states(self, state_seq: torch.Tensor) -> torch.Tensor:
        # state_seq: (B, T, S)
        B, T = state_seq.size(0), state_seq.size(1)
        states = state_seq.view(B * T, -1)
        feats = self.state_encoder(states)
        return feats.view(B, T, -1)

    def encode_obs(
        self, state_seq: Optional[torch.Tensor], image_seq: Optional[torch.Tensor]
    ) -> torch.Tensor:
        parts = []
        if image_seq is not None:
            parts.append(self.encode_images(image_seq))
        if state_seq is not None and self.state_encoder is not None:
            parts.append(self.encode_states(state_seq))

        h = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        h = self.fuse(h)

        # Process entire sequence with MLP
        h = self.sequence_processor(h)  # (B, d_model)
        return self.dropout(h)

    def forward(self, state_seq=None, image_seq=None):
        """Forward pass to predict actions."""
        ctx = self.encode_obs(state_seq, image_seq)
        actions = self.action_head(ctx)
        return actions

    def compute_loss(self, state_seq=None, image_seq=None, actions=None):
        """Compute BC loss (simple MSE)."""
        pred_actions = self.forward(state_seq, image_seq)
        return F.mse_loss(pred_actions, actions)

    @torch.no_grad()
    def sample_actions(
        self,
        state_seq=None,
        image_seq=None,
        steps: int = 32,  # Unused in BC, kept for interface compatibility
        deterministic: bool = False,  # Unused in BC, kept for interface compatibility
    ):
        """Sample/predict actions (deterministic in BC)."""
        return self.forward(state_seq, image_seq)


# Reuse the prepare_batch function from seq_flow
