"""Behavioral Cloning model with RNN and MobileNet components.

This module defines a BC policy using MobileNet for image encoding, MLPs for state encoding,
and LSTM for sequence processing. Designed to have the same input interface as other models
but with RNN-based sequence modeling.
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


class SequenceRNN(nn.Module):
    """LSTM-based sequence processor."""

    def __init__(
        self, feature_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Project LSTM output to desired dimension
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feature_dim)
        B, T, D = x.shape

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last timestep output
        last_output = lstm_out[:, -1, :]  # (B, hidden_dim)

        # Project to output dimension
        output = self.output_proj(last_output)  # (B, out_dim)

        return output


class BCRNNPolicy(nn.Module):
    """RNN-based Behavioral Cloning policy with MobileNet image encoding.

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

        # RNN sequence processor
        rnn_hidden_dim = (
            args.d_model if hasattr(args, "rnn_hidden_dim") else args.d_model
        )
        self.sequence_processor = SequenceRNN(
            feature_dim=args.d_model,
            hidden_dim=rnn_hidden_dim,
            out_dim=args.d_model,
            num_layers=2,
        )

        # Action head - directly predicts actions
        self.action_head = nn.Sequential(
            nn.Linear(args.d_model, args.ff_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.ff_hidden, args.ff_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
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

        # Process sequence with LSTM
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
