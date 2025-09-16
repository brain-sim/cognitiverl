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
from torchvision.models import mobilenet_v3_small, resnet18


class ImageEncoder(nn.Module):
    """MobileNet-based image encoder with ImageNet transforms."""

    def __init__(self, out_dim: int, train: bool = False):
        super().__init__()
        # ImageNet normalization transforms (applied to 0..1 input)
        self.transform_train = transforms.RandomCrop((144, 236))
        self.transform_eval = transforms.Resize((144, 236))
        self.transform_norm = transforms.Normalize(
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
            nn.ELU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        # x: (B, C, H, W) with values in [0, 1]
        # Apply ImageNet normalization
        if train:
            x = self.transform_train(x)
        else:
            x = self.transform_eval(x)
        x = self.transform_norm(x)
        return self.backbone(x)


class ImageEncoderV2(nn.Module):
    """ResNet18 convolutional layers-based image encoder with simple MLP projection."""

    def __init__(self, out_dim: int):
        super().__init__()
        # ImageNet normalization transforms (applied to 0..1 input)
        self.transform_train = transforms.RandomCrop((144, 236))
        self.transform_eval = transforms.Resize((144, 236))
        self.transform_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Load pretrained ResNet18 and extract only conv layers
        resnet = resnet18(pretrained=True)

        # Extract conv layers (everything except avgpool and fc)
        self.resnet_conv_layers = nn.Sequential(*list(resnet.children())[:-2])

        # Global average pooling to reduce spatial dimensions
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Simple MLP projection layer (ResNet18 final conv features = 512)
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        # x: (B, C, H, W) with values in [0, 1]
        # Apply transforms
        if train:
            x = self.transform_train(x)
        else:
            x = self.transform_eval(x)
        x = self.transform_norm(x)

        # Extract conv features using only convolutional layers
        conv_features = self.resnet_conv_layers(x)  # (B, 512, H', W')

        # Global average pooling to get fixed-size features
        pooled_features = self.global_avgpool(conv_features)  # (B, 512, 1, 1)

        # Flatten spatial dimensions
        flattened_features = pooled_features.view(
            pooled_features.size(0), -1
        )  # (B, 512)

        # Project to output dimension
        return self.projection(flattened_features)


class StateEncoder(nn.Module):
    """MLP for encoding state features."""

    def __init__(self, in_dim: int, out_dim: int, ff_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, ff_dim),
            nn.LayerNorm(ff_dim),
            nn.ELU(),
            nn.Linear(ff_dim, 2 * ff_dim),
            nn.LayerNorm(2 * ff_dim),
            nn.ELU(),
            nn.Linear(2 * ff_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceMLP(nn.Module):
    """Simple attention network that processes sequence representations."""

    def __init__(self, seq_len: int, feature_dim: int, out_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.out_dim = out_dim

        # Attention mechanism
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        # Layer norm for attention
        self.ln_attn = nn.LayerNorm(feature_dim)

        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.ELU(),
            nn.Linear(out_dim * 2, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.ELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

        # Scale factor for attention
        self.scale = feature_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feature_dim)
        B, T, D = x.shape

        # Compute attention scores
        q = self.query(x)  # (B, T, feature_dim)
        k = self.key(x)  # (B, T, feature_dim)
        v = self.value(x)  # (B, T, feature_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, T, feature_dim)

        # Add residual connection and layer norm
        attn_output = self.ln_attn(attn_output + x)  # (B, T, feature_dim)

        # Global average pooling to get sequence representation
        sequence_repr = attn_output.mean(dim=1)  # (B, feature_dim)

        # Process with feed-forward network
        return self.ffn(sequence_repr)  # (B, out_dim)


class SequenceTransformer(nn.Module):
    """Transformer encoder that processes sequence representations for BC."""

    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        out_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.out_dim = out_dim

        # Positional embeddings for temporal order
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, feature_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Input projection to ensure proper dimensionality
        self.input_proj = nn.Linear(feature_dim, feature_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Layer norm before pooling
        self.ln_final = nn.LayerNorm(feature_dim)

        # Pooling strategy: learnable weighted average
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feature_dim)
        B, T, D = x.shape

        # Add positional embeddings
        if T <= self.seq_len:
            pos_emb = self.pos_embedding[:, :T, :]
        else:
            # If sequence is longer than expected, interpolate positional embeddings
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=T,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # Project input and add positional embeddings
        x = self.input_proj(x) + pos_emb

        # Apply transformer encoder (bidirectional attention)
        encoded = self.transformer_encoder(x)  # (B, T, feature_dim)

        # Apply final layer norm
        encoded = self.ln_final(encoded)

        # Learnable attention pooling to get sequence representation
        pool_query = self.pool_query.expand(B, -1, -1)  # (B, 1, feature_dim)
        pooled, _ = self.attention_pool(
            query=pool_query, key=encoded, value=encoded
        )  # (B, 1, feature_dim)

        sequence_repr = pooled.squeeze(1)  # (B, feature_dim)

        # Project to output dimension
        return self.output_proj(sequence_repr)  # (B, out_dim)


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

        # Encoders
        d_img = 512
        d_state = 128

        self.state_encoder = StateEncoder(state_dim, d_state) if state_dim > 0 else None
        self.image_encoder = ImageEncoderV2(d_img)

        # Fusion layer
        fusion_input_dim = d_img + (d_state if state_dim > 0 else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fusion_input_dim, args.d_model),
            nn.LayerNorm(args.d_model),
            nn.ELU(),
            nn.Linear(args.d_model, args.d_model),
        )

        # Sequence processor - now using transformer encoder
        self.sequence_processor = SequenceTransformer(
            seq_len=args.sequence_length * args.frame_stack,
            feature_dim=args.d_model,
            out_dim=args.d_model,
            num_heads=8,
            num_layers=3,
            dropout=0.1,
        )

        # Action head - directly predicts actions
        self.action_head = nn.Sequential(
            nn.Linear(args.d_model, args.ff_hidden),
            nn.LayerNorm(args.ff_hidden),
            nn.ELU(),
            nn.Linear(args.ff_hidden, args.ff_hidden),
            nn.LayerNorm(args.ff_hidden),
            nn.ELU(),
            nn.Linear(args.ff_hidden, action_dim),
            nn.Tanh(),  # Output actions in [-1, 1] range
        )

    def encode_images(
        self, image_seq: torch.Tensor, train: bool = False
    ) -> torch.Tensor:
        # image_seq: (B, T, C, H, W)
        B, T = image_seq.size(0), image_seq.size(1)
        images = image_seq.view(B * T, *image_seq.shape[2:]).contiguous(
            memory_format=torch.channels_last
        )
        feats = self.image_encoder(images, train=train)
        return feats.view(B, T, -1)

    def encode_states(self, state_seq: torch.Tensor) -> torch.Tensor:
        # state_seq: (B, T, S)
        B, T = state_seq.size(0), state_seq.size(1)
        states = state_seq.view(B * T, -1)
        feats = self.state_encoder(states)
        return feats.view(B, T, -1)

    def encode_obs(
        self,
        state_seq: Optional[torch.Tensor],
        image_seq: Optional[torch.Tensor],
        train: bool = False,
    ) -> torch.Tensor:
        parts = []

        if image_seq is not None:
            parts.append(self.encode_images(image_seq, train=train))

        if state_seq is not None and self.state_encoder is not None:
            parts.append(self.encode_states(state_seq))

        if len(parts) == 0:
            raise ValueError(
                f"No valid inputs for encoding. "
                f"state_seq={'None' if state_seq is None else state_seq.shape if hasattr(state_seq, 'shape') else type(state_seq)}, "
                f"image_seq={'None' if image_seq is None else image_seq.shape if hasattr(image_seq, 'shape') else type(image_seq)}, "
                f"state_encoder={'None' if self.state_encoder is None else 'available'}"
            )

        h = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        h = self.fuse(h)

        # Process entire sequence with MLP
        h = self.sequence_processor(h)  # (B, d_model)
        return h

    def forward(
        self, state_seq=None, image_seq=None, train: bool = False
    ) -> torch.Tensor:
        """Forward pass to predict actions."""
        ctx = self.encode_obs(state_seq, image_seq, train=train)
        actions = self.action_head(ctx)
        return actions

    def compute_loss(self, state_seq=None, image_seq=None, actions=None):
        """Compute BC loss (simple MSE)."""
        pred_actions = self.forward(state_seq, image_seq, train=True)
        pred_actions = pred_actions + 1e-3 * torch.randn_like(
            pred_actions
        )  # Small noise for stability
        return (
            F.mse_loss(pred_actions, actions, reduction="none").sum(dim=-1).mean()
            + F.l1_loss(pred_actions, actions, reduction="none").sum(dim=-1).mean()
        )

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
