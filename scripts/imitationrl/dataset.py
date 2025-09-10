"""Simple wrapper around robomimic's SequenceDataset with action normalization.

This module provides a minimal wrapper around robomimic's proven SequenceDataset
with just action normalization to [-1, 1] range. No reinventing the wheel!
"""

from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from robomimic.utils import obs_utils as RMObsUtils
from robomimic.utils.dataset import SequenceDataset as RMSequenceDataset

__all__ = ["SequenceDataset", "ActionNormalizer"]


class ActionNormalizer:
    """Reusable action normalizer for [-1, 1] range.

    Can be used both in dataset and environment evaluation.
    Uses the formula: 2 * ((data - min) / (max - min)) - 1
    """

    def __init__(self, action_min: float, action_max: float):
        """Initialize with action min/max values.

        Args:
            action_min: Minimum action value across all dimensions
            action_max: Maximum action value across all dimensions
        """
        self.action_min = float(action_min)
        self.action_max = float(action_max)
        self.action_range = self.action_max - self.action_min + 1e-8

    def normalize(self, actions):
        """Normalize actions to [-1, 1] range.

        Formula: 2 * ((data - min) / (max - min)) - 1

        Args:
            actions: Actions to normalize (numpy array or torch tensor)

        Returns:
            Normalized actions in [-1, 1] range
        """
        normalized = 2 * ((actions - self.action_min) / self.action_range) - 1
        return normalized

    def denormalize(self, normalized_actions):
        """Denormalize actions from [-1, 1] back to original range.

        Args:
            normalized_actions: Normalized actions in [-1, 1] range

        Returns:
            Actions in original range
        """
        actions = ((normalized_actions + 1) / 2) * self.action_range + self.action_min
        return actions

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
        normalize_actions: bool = True,
        modality_type: str = "state+image",  # "state", "image", "state+image"
        **robomimic_kwargs,
    ):
        """Initialize dataset wrapper around robomimic's SequenceDataset.

        Args:
            hdf5_path: Path to HDF5 dataset file
            state_keys: List of state observation keys
            image_keys: List of image observation keys
            sequence_length: Length of sequences
            pad_sequence: Whether to pad sequences
            normalize_actions: Whether to normalize actions to [-1, 1]
            modality_type: Type of modality ("state", "image", "state+image")
            **robomimic_kwargs: Additional arguments passed to robomimic's SequenceDataset
        """
        self.hdf5_path = hdf5_path
        self.modality_type = modality_type
        self.normalize_actions = normalize_actions

        # Default keys
        self.state_keys = state_keys or [
            "left_eef_pos",
            "hand_joint_state",
            "right_eef_pos",
            "right_eef_quat",
            "left_eef_quat",
        ]
        self.image_keys = image_keys or ["robot_pov_cam"]

        # Initialize robomimic's observation utilities
        obs_keys = []
        if modality_type in ("state", "state+image"):
            obs_keys.extend(self.state_keys)
        if modality_type in ("image", "state+image"):
            obs_keys.extend(self.image_keys)

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

        # Create robomimic dataset with defaults + user overrides
        rm_kwargs = {
            "hdf5_path": hdf5_path,
            "obs_keys": tuple(obs_keys),
            "dataset_keys": ("actions",),
            "frame_stack": 1,
            "seq_length": sequence_length,
            "pad_frame_stack": True,
            "pad_seq_length": pad_sequence,
            "get_pad_mask": False,
            "goal_mode": None,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "hdf5_normalize_obs": False,
            "filter_by_attribute": None,
            "load_next_obs": False,
        }
        rm_kwargs.update(robomimic_kwargs)  # Allow user overrides

        self.dataset = RMSequenceDataset(**rm_kwargs)

        # Action normalization
        self.action_normalizer = None
        if normalize_actions:
            action_stats = self._compute_action_stats()
            self.action_normalizer = ActionNormalizer(
                action_stats["min"], action_stats["max"]
            )

    def _compute_action_stats(self) -> Dict[str, float]:
        """Compute action min/max for normalization to [-1, 1]."""
        try:
            # Try to read from file attributes first
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

    def process_batch(
        self, batch: Dict, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Process a batch from robomimic dataset into (state_seq, image_seq, actions).

        Returns:
            state_seq: (B, T, S) or None
            image_seq: (B, T, C, H, W) or None
            actions: (B, A) normalized to [-1, 1]
        """
        # Process actions - take last timestep and normalize
        actions = batch["actions"]
        if actions.dim() == 3:  # (B, T, A) -> (B, A)
            actions = actions[:, -1, :]

        if self.action_normalizer is not None:
            actions = self.action_normalizer.normalize(actions)

        actions = actions.to(device, non_blocking=True)

        # Process observations based on modality
        state_seq = None
        image_seq = None

        if "obs" in batch:
            # States: concatenate requested low-dim keys
            if self.modality_type in ("state", "state+image"):
                state_parts = []
                for k in self.state_keys:
                    if k in batch["obs"]:
                        x = batch["obs"][k]
                        x = x.view(x.shape[0], x.shape[1], -1)  # Flatten last dims
                        state_parts.append(x)

                if state_parts:
                    state_seq = torch.cat(state_parts, dim=-1)
                    state_seq = state_seq.to(device, non_blocking=True)

            # Images: use first key (can be extended for multi-camera)
            if self.modality_type in ("image", "state+image"):
                if len(self.image_keys) > 0:
                    key = self.image_keys[0]
                    if key in batch["obs"]:
                        img = batch["obs"][key]

                        # Convert to BTCHW if needed
                        if img.dim() == 5 and img.shape[-1] in (
                            1,
                            3,
                            4,
                        ):  # B,T,H,W,C → B,T,C,H,W
                            img = img.permute(0, 1, 4, 2, 3)

                        img = img.float()

                        # Normalize to [0,1] if needed
                        if img.max() > 1.0:
                            img = (img / 255.0).clamp(0.0, 1.0)

                        image_seq = img.to(device, non_blocking=True)

        return state_seq, image_seq, actions

    def __len__(self) -> int:
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
    print(
        "✅ Robomimic wrapper ready! Use robomimic's proven SequenceDataset with action normalization."
    )
    print("Example usage:")
    dataset = SequenceDataset(
        hdf5_path="/home/chandramouli/cognitiverl/datasets/generated_dataset_gr1_nut_pouring.hdf5",
        state_keys=[
            "left_eef_pos",
            "hand_joint_state",
            "right_eef_pos",
            "right_eef_quat",
            "left_eef_quat",
        ],
        image_keys=["robot_pov_cam"],
        sequence_length=10,
        normalize_actions=True,
    )

    # Get action normalizer for environment evaluation
    normalizer = dataset.get_action_normalizer()

    # Use in training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = dataset.get_dataloader(batch_size=32)
    for batch in dataloader:
        state_seq, image_seq, actions = dataset.process_batch(batch, device)
        print(
            state_seq.shape,
            state_seq.dtype,
            state_seq.device,
            state_seq.max(),
            state_seq.min(),
        )
        print(
            image_seq.shape,
            image_seq.dtype,
            image_seq.device,
            image_seq.max(),
            image_seq.min(),
        )
        print(
            actions.shape, actions.dtype, actions.device, actions.max(), actions.min()
        )
        break
    # actions are now in [-1, 1] range


if __name__ == "__main__":
    test_simple_dataset()
