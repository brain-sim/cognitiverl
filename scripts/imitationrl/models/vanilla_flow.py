"""Vanilla Flow-BC model with MobileNet and MLP components.

This module defines a simplified flow-matching policy using MobileNet for image encoding
and MLPs for processing, without transformer components. Designed to have the same input
interface as SeqFlowPolicy but with a vanilla MLP-based architecture.
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
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
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
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
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
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feature_dim)
        B, T, D = x.shape

        # Flatten sequence dimension
        x_flat = x.view(B, T * D)  # (B, T * feature_dim)

        # Process with MLP
        return self.net(x_flat)


class VanillaFlowPolicy(nn.Module):
    """MLP-based flow-matching policy with MobileNet image encoding.

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
        d_img = args.d_model // 2
        d_state = args.d_model // 2

        self.state_encoder = StateMLP(state_dim, d_state) if state_dim > 0 else None
        self.image_encoder = ImageEncoder(d_img)

        # Fusion layer
        fusion_input_dim = d_img + (d_state if state_dim > 0 else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fusion_input_dim, args.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model, args.d_model),
        )

        # Sequence processor - processes entire sequence with MLP
        self.sequence_processor = SequenceMLP(
            seq_len=args.sequence_length, feature_dim=args.d_model, out_dim=args.d_model
        )

        # Context processor (additional processing after sequence)
        self.context_processor = nn.Sequential(
            nn.Linear(args.d_model, args.ff_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.ff_hidden, args.d_model),
            nn.ReLU(inplace=True),
        )

        # Flow network: predicts velocity given context, x_t, t
        self.flow = nn.Sequential(
            nn.Linear(args.d_model + action_dim + 1, args.ff_hidden),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.ff_hidden, args.ff_hidden),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.ff_hidden, action_dim),
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

        # Process entire sequence with MLP (instead of taking last timestep)
        h = self.sequence_processor(h)  # (B, d_model)

        # Additional context processing
        ctx = self.context_processor(h)
        return self.dropout(ctx)

    def compute_flow_loss(self, state_seq=None, image_seq=None, actions=None):
        B = actions.size(0)
        ctx = self.encode_obs(state_seq, image_seq)

        t = torch.rand(B, 1, device=actions.device)
        noise = torch.randn_like(actions)
        x_t = (1 - t) * noise + t * actions
        v_target = actions - noise

        inp = torch.cat([ctx, x_t, t], dim=-1)
        v_pred = self.flow(inp)
        return F.mse_loss(v_pred, v_target, reduction="none").sum(dim=-1).mean()

    @torch.no_grad()
    def sample_actions(
        self,
        state_seq=None,
        image_seq=None,
        steps: int = 32,
        deterministic: bool = False,
    ):
        B = state_seq.size(0) if state_seq is not None else image_seq.size(0)
        ctx = self.encode_obs(state_seq, image_seq)

        dt = 1.0 / steps
        if deterministic:
            x = torch.zeros(B, self.action_dim, device=ctx.device)
        else:
            x = torch.randn(B, self.action_dim, device=ctx.device)
        for i in range(steps):
            t = torch.full((B, 1), i * dt, device=ctx.device)
            v = self.flow(torch.cat([ctx, x, t], dim=-1))
            x = x + dt * v
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x


# Reuse the prepare_batch function from seq_flow
