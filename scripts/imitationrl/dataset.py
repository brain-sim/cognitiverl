"""Simple wrapper around robomimic's SequenceDataset with action normalization.

This module provides a minimal wrapper around robomimic's proven SequenceDataset
with just action normalization to [-1, 1] range. No reinventing the wheel!
"""

import copy
import os
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from robomimic.utils import obs_utils as RMObsUtils
from robomimic.utils.dataset import SequenceDataset as RMSequenceDataset
from tensordict import TensorDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm

from scripts.imitationrl.utils import make_image_grid, process_image_batch

__all__ = ["SequenceDataset", "ActionNormalizer"]


class ActionNormalizer:
    """Reusable action normalizer for [-1, 1] range.

    Can be used both in dataset and environment evaluation.
    Uses the formula: 2 * ((data - min) / (max - min)) - 1
    """

    def __init__(self, action_min: float, action_max: float):
        """Initialize with action min/max values."""
        self.action_min = float(action_min)
        self.action_max = float(action_max)
        self.action_range = self.action_max - self.action_min + 1e-8

    def normalize(self, actions):
        """Normalize actions to [-1, 1] range."""
        return 2 * ((actions - self.action_min) / self.action_range) - 1

    def denormalize(self, normalized_actions):
        """Denormalize actions from [-1, 1] back to original range."""
        return ((normalized_actions + 1) / 2) * self.action_range + self.action_min

    def get_stats(self):
        """Get the action statistics as a dict."""
        return {"min": self.action_min, "max": self.action_max}

    @classmethod
    def from_stats(cls, stats_dict):
        """Create normalizer from stats dictionary."""
        return cls(stats_dict["min"], stats_dict["max"])


class SequenceDataset:
    """Simple wrapper around robomimic's SequenceDataset with action normalization."""

    def __init__(
        self,
        hdf5_path: str,
        state_keys: List[str] = None,
        image_keys: List[str] = None,
        sequence_length: int = 10,
        pad_sequence: bool = True,
        pad_frame_stack: bool = True,
        normalize_actions: bool = True,
        modality_type: str = "state+image",  # "state", "image", "state+image"
        demo_limit: int | None = None,
        frame_stack: int = 1,
        **robomimic_kwargs,
    ):
        """Initialize dataset wrapper around robomimic's SequenceDataset."""
        self.hdf5_path = hdf5_path
        self.modality_type = modality_type
        self.normalize_actions = normalize_actions
        self.demo_limit = demo_limit
        self.frame_stack = frame_stack
        self.sequence_length = sequence_length
        self.pad_frame_stack = pad_frame_stack
        self.pad_sequence = pad_sequence
        # Default keys - order matters for state concatenation
        print(f"🔑 Using state_keys: {state_keys}")
        self.state_keys = (
            [
                "left_eef_pos",
                "left_eef_quat",
                "right_eef_pos",
                "right_eef_quat",
                "hand_joint_state",
            ]
            if state_keys is None
            else state_keys
        )
        print(f"🔑 Using state_keys: {self.state_keys}")
        self.image_keys = image_keys or ["robot_pov_cam"]
        print(f"🖼️  Using image_key: {self.image_keys}")

        # Setup observation keys based on modality
        obs_keys = []
        if modality_type in ("state", "state+image"):
            obs_keys.extend(self.state_keys)
        if modality_type in ("image", "state+image"):
            obs_keys.extend(self.image_keys)

        # Initialize robomimic's observation utilities
        RMObsUtils.initialize_obs_utils_with_obs_specs(
            {
                "obs": {
                    "low_dim": self.state_keys
                    if modality_type in ("state", "state+image")
                    else [],
                    "rgb": self.image_keys
                    if modality_type in ("image", "state+image")
                    else [],
                }
            }
        )

        # Create robomimic dataset with sensible defaults
        rm_kwargs = {
            "hdf5_path": hdf5_path,
            "obs_keys": tuple(obs_keys),
            "dataset_keys": ("actions",),
            "frame_stack": self.frame_stack,
            "seq_length": self.sequence_length,  # Force seq_length to 1 when using frame_stack as time
            "pad_seq_length": self.pad_sequence,
            "pad_frame_stack": self.pad_frame_stack,
            "get_pad_mask": False,
            "goal_mode": None,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": False,
            "hdf5_normalize_obs": False,
            "filter_by_attribute": None,
            "load_next_obs": False,
        }
        rm_kwargs.update(robomimic_kwargs)  # Allow user overrides

        self.dataset = RMSequenceDataset(**rm_kwargs)

        # Setup action normalization
        self.action_normalizer = None
        if normalize_actions:
            action_stats = self._compute_action_stats()
            self.action_normalizer = ActionNormalizer(
                action_stats["min"], action_stats["max"]
            )

    def _compute_action_stats(self) -> Dict[str, float]:
        """Compute action min/max for normalization to [-1, 1]."""
        try:
            # Try reading from file attributes first
            with h5py.File(self.hdf5_path, "r") as f:
                amin = f.attrs.get("action_min", None)
                amax = f.attrs.get("action_max", None)
                if amin is not None and amax is not None:
                    return {"min": float(amin), "max": float(amax)}
        except Exception:
            pass

        # Compute from data
        min_val, max_val = np.inf, -np.inf
        with h5py.File(self.hdf5_path, "r") as f:
            for demo_id in sorted(f["data"].keys(), key=lambda x: int(x[5:])):
                actions = np.asarray(f[f"data/{demo_id}/actions"])
                if actions.size > 0:
                    min_val = min(min_val, float(np.min(actions)))
                    max_val = max(max_val, float(np.max(actions)))

        if not np.isfinite(min_val) or not np.isfinite(max_val):
            min_val, max_val = -1.0, 1.0

        return {"min": float(min_val), "max": float(max_val)}

    def get_action_normalizer(self):
        """Get the action normalizer for use in environment evaluation."""
        return self.action_normalizer

    def denormalize_actions(self, normalized_actions):
        """Denormalize actions from [-1, 1] back to original range."""
        if self.action_normalizer is None:
            return normalized_actions
        return self.action_normalizer.denormalize(normalized_actions)

    def process_batch(self, batch: Dict, device: torch.device) -> TensorDict:
        """Process a batch from robomimic dataset into a clean TensorDict.

        Returns:
            TensorDict with keys: 'state_seq', 'image_seq', 'actions'
            - state_seq: (B, T, S) or None if not using state modality
            - image_seq: (B, T, C, H, W) or None if not using image modality
            - actions: (B, A) normalized to [-1, 1]
        """
        batch_size = None
        result_dict = {}

        # Process actions - take last timestep and normalize
        actions = batch["actions"]
        if actions.dim() == 3:  # (B, T, A) -> (B, A)
            actions = actions[:, -1, :]

        if self.action_normalizer is not None:
            actions = self.action_normalizer.normalize(actions)

        result_dict["actions"] = actions.to(device, non_blocking=True)
        batch_size = actions.shape[0]

        # Process observations based on modality
        if "obs" in batch:
            # Process states: concatenate in exact order of state_keys
            if self.modality_type in ("state", "state+image"):
                state_tensors = []
                for key in self.state_keys:  # Maintain exact order
                    if key in batch["obs"]:
                        state_data = batch["obs"][key]
                        # Flatten last dimensions: (B, T, ...) -> (B, T, -1)
                        state_data = state_data.view(
                            state_data.shape[0], state_data.shape[1], -1
                        )
                        state_tensors.append(state_data)

                if state_tensors:
                    state_seq = torch.cat(state_tensors, dim=-1)
                    result_dict["state_seq"] = state_seq.to(device, non_blocking=True)
                else:
                    result_dict["state_seq"] = None
            else:
                result_dict["state_seq"] = None

            # Process images: use first image key
            if self.modality_type in ("image", "state+image"):
                image_seq = None
                if self.image_keys and self.image_keys[0] in batch["obs"]:
                    img = batch["obs"][self.image_keys[0]]

                    # Use shared image processing function
                    image_seq = process_image_batch(
                        img, target_format="BTCHW", normalize_to_01=True, device=device
                    )

                result_dict["image_seq"] = image_seq
            else:
                result_dict["image_seq"] = None
        else:
            result_dict["state_seq"] = None
            result_dict["image_seq"] = None

        # Create TensorDict with proper batch size
        return TensorDict(result_dict, batch_size=[batch_size])

    def __len__(self) -> int:
        if self.demo_limit is not None:
            return min(len(self.dataset), self.demo_limit * 300)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """Get a DataLoader for this dataset."""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,  # Drop last for training, keep for validation
            **kwargs,
        )


def test_simple_dataset():
    """Simple test for the dataset wrapper."""
    print("✅ Robomimic wrapper ready! Using TensorDict for clean batch processing.")
    print("Example usage:")

    dataset = SequenceDataset(
        hdf5_path="/home/chandramouli/cognitiverl/datasets/generated_dataset_gr1_nut_pouring.hdf5",
        state_keys=[
            "left_eef_pos",
            "left_eef_quat",
            "right_eef_pos",
            "right_eef_quat",
            "hand_joint_state",
        ],
        image_keys=["robot_pov_cam"],
        sequence_length=10,
        frame_stack=1,
        normalize_actions=False,
        pad_sequence=True,
        pad_frame_stack=True,
    )

    # Get action normalizer for environment evaluation
    normalizer = dataset.get_action_normalizer()

    # Use in training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = dataset.get_dataloader(batch_size=32)

    for batch in dataloader:
        tensor_dict = dataset.process_batch(batch, device)

        print("TensorDict keys:", list(tensor_dict.keys()))
        print("Batch size:", tensor_dict.batch_size)

        if tensor_dict["state_seq"] is not None:
            state_seq = tensor_dict["state_seq"]
            print(f"State: {state_seq.shape}, {state_seq.dtype}, {state_seq.device}")
            print(f"State range: [{state_seq.min():.3f}, {state_seq.max():.3f}]")

        if tensor_dict["image_seq"] is not None:
            image_seq = tensor_dict["image_seq"]
            print(f"Image: {image_seq.shape}, {image_seq.dtype}, {image_seq.device}")
            print(f"Image range: [{image_seq.min():.3f}, {image_seq.max():.3f}]")

        actions = tensor_dict["actions"]
        print(f"Actions: {actions.shape}, {actions.dtype}, {actions.device}")
        print(f"Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
        break

    print("✅ Actions are normalized to [-1, 1] range")
    print("✅ State concatenation preserves exact key order")
    print("✅ TensorDict provides clean batch structure")


def test_debug_dataset_samples():
    """Sample 5 items via __getitem__, print states and denormalized actions, and save image grids.

    - Prints concatenated state sequences (per sample) and denormalized actions
    - Collects image sequences, builds a grid using make_image_grid, and saves to
      /home/chandramouli/cogntiverl/wandb/dataset_debug
    """

    # ---- Config ----
    hdf5_path = (
        "/home/chandramouli/cognitiverl/datasets/generated_dataset_gr1_nut_pouring.hdf5"
    )
    debug_dir = "/home/chandramouli/cognitiverl/wandb/dataset_debug"
    os.makedirs(debug_dir, exist_ok=True)

    dataset = SequenceDataset(
        hdf5_path=hdf5_path,
        state_keys=[
            "left_eef_pos",
            "left_eef_quat",
            "right_eef_pos",
            "right_eef_quat",
            "hand_joint_state",
        ],
        image_keys=["robot_pov_cam"],
        sequence_length=1,
        frame_stack=1,
        pad_sequence=True,
        pad_frame_stack=True,
        normalize_actions=False,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        print(batch["obs"]["robot_pov_cam"].shape)
        print(batch["obs"]["left_eef_pos"].shape)
        print(batch["actions"].shape)
        return

    n = len(dataset)
    if n == 0:
        print("Dataset is empty.")
        return

    # Choose 5 indices spread across dataset
    if n < 5:
        sample_indices = list(range(n))
    else:
        sample_indices = [
            0,
            max(1, n // 4),
            max(2, n // 2),
            max(3, (3 * n) // 4),
            n - 1,
        ]

    print("\n=== Debugging 5 samples from __getitem__ ===")
    image_batches = []  # Will hold (1, T, C, H, W) per sample
    sample_indices = [10]
    for i, idx in enumerate(sample_indices):
        item = dataset[idx]

        # ---- Actions (denormalized) ----
        actions = item.get("actions")
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.float()
        print(f"Action shape: {actions.shape}")
        print(f"Action: {actions[:, 3:7]}")

        # If action has time/sequence dims, take last step for readability
        if actions.dim() > 1:
            actions_for_print = actions[-1]
        else:
            actions_for_print = actions

        # If actions look already normalized to [-1, 1], denormalize them; else keep as-is
        a_min, a_max = actions_for_print.min().item(), actions_for_print.max().item()
        if dataset.action_normalizer is not None and a_min >= -1.05 and a_max <= 1.05:
            actions_denorm = dataset.denormalize_actions(actions_for_print)
        else:
            actions_denorm = actions_for_print

        # ---- State sequence (concatenate in the configured order) ----
        state_seq = None
        if "obs" in item:
            state_tensors = []
            for key in dataset.state_keys:
                if key in item["obs"]:
                    s = item["obs"][key]
                    if isinstance(s, np.ndarray):
                        s = torch.from_numpy(s)
                    s = s.float()
                    # Ensure shape (T, -1)
                    if s.dim() == 1:
                        s = s.unsqueeze(0)
                    s = s.view(s.shape[0], -1)
                    state_tensors.append(s)
            if state_tensors:
                state_seq = torch.cat(state_tensors, dim=-1)
        print(f"State: {state_seq[:, 3:7]}")

        # ---- Print debug info ----

        last_state = state_seq[0].cpu().numpy() if state_seq is not None else None
        actions_np = actions_denorm.cpu().numpy()

        if last_state is not None and last_state.shape == actions_np.shape:
            # Define the index groups and their names
            groups = [
                ("right_eef_quat", 10, 14),
            ]

            print("\n🟢✨ State vs. Actions (Denormalized) Comparison Table ✨🟢")
            print("Legend: Δ = Action (denorm) - State (last step)\n")

            for group_name, start_idx, end_idx in groups:
                s_group = last_state[start_idx:end_idx]
                a_group = actions_np[start_idx:end_idx]
                diff_group = a_group - s_group

                print(f"--- {group_name} (indices {start_idx}-{end_idx - 1}) ---")
                print(
                    "┌────┬──────────────────────┬──────────────────────┬──────────────────────┐"
                )
                print(
                    "│ Id │   State (last step)  │   Action (denorm)    │   Δ (Action-State)   │"
                )
                print(
                    "├────┼──────────────────────┼──────────────────────┼──────────────────────┤"
                )
                for j in range(end_idx - start_idx):
                    s = s_group[j]
                    a = a_group[j]
                    diff = diff_group[j]
                    print(
                        f"│{j + start_idx:>3} │ {s:>18.5f}     │ {a:>18.5f}     │ {diff:>18.5f}     │"
                    )
                print(
                    "└────┴──────────────────────┴──────────────────────┴──────────────────────┘"
                )
        else:
            print("\n🟠 actions (denormalized) breakdown:")
            print(" ".join([f"{x:.4f}" for x in actions_np]))
            print("Full actions (denorm):", actions_np)

        # ---- Images ----
        if "obs" in item and dataset.image_keys:
            img_key = dataset.image_keys[0]
            if img_key in item["obs"]:
                img = item["obs"][img_key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                img = img.float()

                # Ensure shape (1, T, C, H, W) for make_image_grid
                if img.dim() == 4:  # (T, H, W, C) or (T, C, H, W)
                    img = img.unsqueeze(0)
                elif img.dim() == 3:  # (H, W, C) or (C, H, W) – add T and B dims
                    img = img.unsqueeze(0).unsqueeze(0)

                img_btchw = process_image_batch(
                    img, target_format="BTCHW", normalize_to_01=True
                )
                image_batches.append(img_btchw)

                # Also save per-sample grid
                sample_grid = make_image_grid(img_btchw, nrow=1, max_images=10)
                sample_path = os.path.join(debug_dir, f"sample_{i + 1}_idx_{idx}.png")
                torchvision.utils.save_image(sample_grid, sample_path)
                print(f"saved sample grid: {sample_path}")
                print(
                    f"🔴 Difference between images along the time dimension: {torch.diff(img_btchw, dim=1).mean()}"
                )
            else:
                print(f"image key '{img_key}' not found in obs")
        else:
            print("no images found in obs")

    # ---- Save combined grid for all collected samples ----
    if image_batches:
        all_imgs = torch.cat(image_batches, dim=0)  # (B, T, C, H, W)
        grid = make_image_grid(all_imgs, nrow=5, max_images=100)
        grid_path = os.path.join(debug_dir, "samples_grid.png")
        torchvision.utils.save_image(grid, grid_path)
        print(f"\nCombined samples grid saved to: {grid_path}")
    else:
        print("\nNo images collected to build a grid.")


def train_vqvae(dataset: torch.utils.data.Dataset, epochs: int = 100):
    """
    Train a very simple VQ-VAE (encoder + vector quantizer + decoder) only on
    the image modality, overfitting the first 300 samples. Saves per-epoch
    reconstruction and ground truth grids to:
    - /home/chandramouli/cognitiverl/wandb/dataset_debug/recons_epoch_{idx}.png
    - /home/chandramouli/cognitiverl/wandb/dataset_debug/gt_epoch_{idx}.png

    Returns the trained VQ-VAE model.
    """

    debug_dir = "/home/chandramouli/cognitiverl/wandb/dataset_debug"
    os.makedirs(debug_dir, exist_ok=True)

    class VectorQuantizer(nn.Module):
        def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.beta = beta
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.uniform_(
                self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings
            )

        def forward(self, z_e):
            # z_e: (B, C, H, W)
            B, C, H, W = z_e.shape
            flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)
            emb = self.embedding.weight  # (K, C)
            # Squared Euclidean distances: ||z - e||^2 = ||z||^2 - 2 z·e + ||e||^2
            distances = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat @ emb.t()
                + emb.pow(2).sum(dim=1)
            )  # (B*H*W, K)
            indices = distances.argmin(dim=1)
            z_q = (
                self.embedding(indices)
                .view(B, H, W, C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            # VQ losses
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commitment_loss = 0.25 * F.mse_loss(z_e, z_q.detach())
            vq_loss = codebook_loss + commitment_loss

            # Straight-through estimator: passthrough gradients to encoder
            z_q_st = z_e + (z_q - z_e).detach()
            return z_q_st, vq_loss

    class Encoder(nn.Module):
        def __init__(self, embedding_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(
                    3, 32, kernel_size=4, stride=2, padding=1
                ),  # 160x256 -> 80x128
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    32, 64, kernel_size=4, stride=2, padding=1
                ),  # 80x128 -> 40x64
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    64, 128, kernel_size=4, stride=2, padding=1
                ),  # 40x64 -> 20x32
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            )

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self, embedding_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    128, 64, kernel_size=4, stride=2, padding=1
                ),  # 20x32 -> 40x64
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    64, 32, kernel_size=4, stride=2, padding=1
                ),  # 40x64 -> 80x128
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    32, 3, kernel_size=4, stride=2, padding=1
                ),  # 80x128 -> 160x256
                nn.Sigmoid(),  # keep outputs in [0, 1]
            )

        def forward(self, z):
            return self.net(z)

    class VQVAE(nn.Module):
        def __init__(self, codebook_size: int = 256, embedding_dim: int = 64):
            super().__init__()
            self.encoder = Encoder(embedding_dim)
            self.quantizer = VectorQuantizer(codebook_size, embedding_dim)
            self.decoder = Decoder(embedding_dim)

        def forward(self, x):
            z_e = self.encoder(x)
            z_q, vq_loss = self.quantizer(z_e)
            x_recon = self.decoder(z_q)
            return x_recon, vq_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(codebook_size=256, embedding_dim=64).to(device)

    # Overfit on first 300 samples
    n_overfit = min(len(dataset), 300)
    subset = torch.utils.data.Subset(dataset, list(range(n_overfit)))
    dataloader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    pbar = tqdm(range(epochs), desc="VQ-VAE (images) overfitting 300")

    last_recon = None
    last_gt = None
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            obs = batch["obs"]
            img = obs["robot_pov_cam"][:, 0]  # (B, H, W, C)
            img = img.permute(0, 3, 1, 2).float().to(device) / 255.0  # (B, C, H, W)

            x_recon, vq_loss = model(img)
            recon_loss = F.mse_loss(x_recon, img)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            last_recon = x_recon.detach().cpu()
            last_gt = img.detach().cpu()

        avg_loss = epoch_loss / max(1, len(dataloader))
        pbar.set_postfix(loss=avg_loss)

        # Save last batch grids this epoch
        if last_recon is not None and last_gt is not None:
            grid_recon = torchvision.utils.make_grid(
                last_recon, nrow=min(8, last_recon.size(0))
            )
            grid_gt = torchvision.utils.make_grid(last_gt, nrow=min(8, last_gt.size(0)))
            recon_path = os.path.join(debug_dir, f"recons_epoch_{epoch}.png")
            gt_path = os.path.join(debug_dir, f"gt_epoch_{epoch}.png")
            torchvision.utils.save_image(grid_recon, recon_path)
            torchvision.utils.save_image(grid_gt, gt_path)

    model.eval()
    return model


def train_dataset(dataset: torch.utils.data.Dataset, epochs: int = 100):
    r"""
    Train a given state + image dataset using behavior cloning.
    Create a simple model with:
    image -------- state
     |              |
    CNN encoder -- state encoder
       \         /
            MLP
            |
          action
    I want to overfit the model to the first 300 samples.
    Define the model architecture a simple one with layernorm
    and elu activation inside the function as nested class.
    Create a simple data preprocessing step with barebone working for image + state modality only.
    Make sure that only the image is normalized to [0, 1].
    Train the model with a progress bar showing loss progress
    and return the trained model.
    """

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_encoder = nn.Sequential(
                nn.Linear(36, 64),
                nn.ELU(),
                nn.LayerNorm(64),
            )
            self.action_encoder = nn.Sequential(
                nn.Linear(36, 64),
                nn.ELU(),
                nn.LayerNorm(64),
            )
            self.bc_transformer = nn.Transformer(
                d_model=64,
                nhead=8,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=256,
                batch_first=True,
            )
            self.bc_classifier = nn.Sequential(
                # Transformer outputs d_model=64 along the feature dim
                nn.Linear(64, 36),
                nn.ELU(),
                nn.LayerNorm(36),
                nn.Linear(36, 36),
            )

        def forward(self, image, state, action=None):
            x = self.state_encoder(state)
            a = torch.zeros_like(x[:, 0]).unsqueeze(1)
            x = self.bc_transformer(x, tgt=a)
            action_pred = self.bc_classifier(x)
            return action_pred[:, 0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    # Overfit on first 300 samples
    n_overfit = min(len(dataset), 30000)
    subset = torch.utils.data.Subset(dataset, list(range(n_overfit)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    # Drop LR by 10x after 100 epochs to help final convergence
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200], gamma=0.1
    )
    best_model = copy.deepcopy(model)
    best_loss = float("inf")
    progress_bar = tqdm(
        range(epochs), desc="Training (overfitting 300 samples)", position=0
    )

    dataloader = DataLoader(
        subset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        pin_memory_device="cuda",  # requires torch>=2.0
        persistent_workers=True,
        prefetch_factor=4,  # try 4–6 if I/O is fast
        multiprocessing_context="spawn",
    )
    dataloader_progress_bar = tqdm(range(len(dataloader)), desc="Progress", position=1)

    for epoch in progress_bar:
        losses = 0
        dataloader_progress_bar.reset()
        for batch in dataloader:
            obs, action = batch["obs"], batch["actions"].float().to(device)
            state = (
                torch.cat(
                    [
                        obs["left_eef_pos"],
                        obs["left_eef_quat"],
                        obs["right_eef_pos"],
                        obs["right_eef_quat"],
                        obs["hand_joint_state"],
                    ],
                    dim=-1,
                )
                .float()
                .to(device)
            )
            action_pred = model(None, state, action)
            loss = (
                F.mse_loss(action_pred, action[:, -1], reduction="none")
                .sum(dim=-1)
                .mean()
                + F.l1_loss(action_pred, action[:, -1], reduction="none")
                .sum(dim=-1)
                .mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            dataloader_progress_bar.update(1)
            dataloader_progress_bar.set_postfix(step_loss=loss.item())
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix(loss=losses / len(dataloader), lr=current_lr)
        if losses / len(dataloader) < best_loss:
            best_loss = losses / len(dataloader)
            best_model = copy.deepcopy(model)
    best_model.eval()
    return best_model


def plug_and_play_dataset():
    # ---- Config ----
    hdf5_path = (
        "/home/chandramouli/cognitiverl/datasets/generated_dataset_gr1_nut_pouring.hdf5"
    )
    debug_dir = "/home/chandramouli/cognitiverl/wandb/dataset_debug"
    os.makedirs(debug_dir, exist_ok=True)

    dataset = SequenceDataset(
        hdf5_path=hdf5_path,
        state_keys=[
            "left_eef_pos",
            "left_eef_quat",
            "right_eef_pos",
            "right_eef_quat",
            "hand_joint_state",
        ],
        image_keys=["robot_pov_cam"],
        sequence_length=1,
        frame_stack=10,
        pad_sequence=True,
        pad_frame_stack=True,
        normalize_actions=False,
    )
    trained_model = train_dataset(dataset, epochs=250)
    trained_model.eval()
    from argparse import Namespace

    import pinocchio  # noqa: F401
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        Namespace(
            **{
                "headless": True,
                "enable_cameras": True,
            }
        )
    )
    simulation_app = app_launcher.app
    import isaaclab_tasks  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

    from scripts.utils import make_isaaclab_env

    env = make_isaaclab_env(
        seed=1,
        task="Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
        device="cuda:0",
        num_envs=1,
        capture_video=False,
        disable_fabric=False,
    )()
    device = torch.device("cuda:0")

    obs, _ = env.reset()
    state = torch.cat(
        [
            obs["left_eef_pos"],
            obs["left_eef_quat"],
            obs["right_eef_pos"],
            obs["right_eef_quat"],
            obs["hand_joint_state"],
        ],
        dim=-1,
    ).to(device)
    state_buffer = state.unsqueeze(1).expand(1, 10, -1)
    image = obs["robot_pov_cam"].permute(0, 3, 1, 2).to(device) / 255.0
    steps = 300
    image_buffer = torch.zeros(1, steps, 3, 160, 256, dtype=torch.float32)
    image_buffer[:, 0] = image.cpu()

    for i in range(1, steps):
        action = torch.from_numpy(dataset[i]["actions"]).to(device)
        action_pred = trained_model(None, state_buffer)
        obs, _, _, _ = env.step(action_pred.detach())
        state = torch.cat(
            [
                obs["left_eef_pos"],
                obs["left_eef_quat"],
                obs["right_eef_pos"],
                obs["right_eef_quat"],
                obs["hand_joint_state"],
            ],
            dim=-1,
        ).to(device)
        state_buffer = torch.roll(state_buffer, shifts=-1, dims=1)
        state_buffer[:, -1] = state
        image = obs["robot_pov_cam"].permute(0, 3, 1, 2).to(device) / 255.0
        image_buffer[:, i] = image.cpu()

    image_buffer = image_buffer.reshape(steps // 10, 10, 3, 160, 256)
    grid_image = make_image_grid(image_buffer, nrow=10, max_images=steps)
    torchvision.utils.save_image(
        grid_image,
        "/home/chandramouli/cognitiverl/wandb/dataset_debug/image_pred_buffer.png",
    )
    env.close()
    return simulation_app


if __name__ == "__main__":
    simulation_app = None
    try:
        simulation_app = plug_and_play_dataset()
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print(e)
    if simulation_app:
        simulation_app.close()
