"""Sequence Flow-BC model components.

This module defines lightweight encoders and a Transformer-based policy for
sequence-conditioned behavioral cloning using flow matching. It intentionally
excludes training and logging code — see `scripts/imitation_learning/train.py`
for the end-to-end training loop.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    """Light image encoder: (C,H,W) -> d_model_img."""

    def __init__(self, in_ch: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        h = self.net(x).flatten(1)
        return self.proj(h)


class StateMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class SeqFlowPolicy(nn.Module):
    """Transformer-based flow-matching policy conditioned on a sequence.

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
        self.image_encoder = TinyCNN(args.img_size[0], d_img)
        self.fuse = nn.Linear(d_img + (d_state if state_dim > 0 else 0), args.d_model)

        # Sequence model
        self.posenc = PositionalEncoding(args.d_model, max_len=args.sequence_length)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.ff_hidden,
            dropout=args.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=args.num_layers)

        # Flow network: predicts velocity given context, x_t, t
        self.flow = nn.Sequential(
            nn.Linear(args.d_model + action_dim + 1, args.ff_hidden),
            nn.SiLU(),
            nn.Linear(args.ff_hidden, args.ff_hidden),
            nn.SiLU(),
            nn.Linear(args.ff_hidden, action_dim),
        )

    def encode_images(self, image_seq: torch.Tensor) -> torch.Tensor:
        # image_seq: (B, T, C, H, W)
        B, T = image_seq.size(0), image_seq.size(1)
        images = image_seq.view(B * T, *image_seq.shape[2:])
        try:
            images = images.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass
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
        h = self.posenc(h)
        h = self.transformer(h)
        # Use representation at the last timestep as context
        ctx = h[:, -1, :]
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
    def sample_actions(self, state_seq=None, image_seq=None, steps: int = 32):
        B = state_seq.size(0) if state_seq is not None else image_seq.size(0)
        ctx = self.encode_obs(state_seq, image_seq)

        dt = 1.0 / steps
        x = torch.randn(B, self.action_dim, device=ctx.device)
        for i in range(steps):
            t = torch.full((B, 1), i * dt, device=ctx.device)
            v = self.flow(torch.cat([ctx, x, t], dim=-1))
            x = x + dt * v
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x

def prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device, modality: str):
    """Move batch to device and select the proper tensors.

    Expects:
    - batch["actions"]: (B, T, A) or (B, A); returns (B, A) current action.
    - batch["state_obs"]: (B, T, S) if present
    - batch["image_obs"]: (B, T, C, H, W) or dict of such
    """
    actions = batch["actions"].to(device, non_blocking=True)
    if actions.dim() == 3:
        actions = actions[:, -1, :]

    state_seq = None
    image_seq = None
    if modality in ("state", "state+image") and "state_obs" in batch:
        state_seq = batch["state_obs"].to(device, non_blocking=True)
    if modality in ("image", "state+image") and "image_obs" in batch:
        img = batch["image_obs"]
        if isinstance(img, dict):
            first_key = sorted(img.keys())[0]
            img = img[first_key]
        image_seq = img.to(device, non_blocking=True)

    return state_seq, image_seq, actions
