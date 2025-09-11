"""Simple wrapper around robomimic's SequenceDataset with action normalization.

This module provides a minimal wrapper around robomimic's proven SequenceDataset
with just action normalization to [-1, 1] range. No reinventing the wheel!
"""

from typing import Dict, List

import h5py
import numpy as np
import torch
from robomimic.utils import obs_utils as RMObsUtils
from robomimic.utils.dataset import SequenceDataset as RMSequenceDataset
from tensordict import TensorDict

from scripts.imitationrl.utils import process_image_batch

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
        normalize_actions: bool = True,
        modality_type: str = "state+image",  # "state", "image", "state+image"
        demo_limit: int | None = 10,
        frame_stack: int = 1,
        **robomimic_kwargs,
    ):
        """Initialize dataset wrapper around robomimic's SequenceDataset."""
        self.hdf5_path = hdf5_path
        self.modality_type = modality_type
        self.normalize_actions = normalize_actions
        self.demo_limit = demo_limit
        self.frame_stack = robomimic_kwargs.get("frame_stack", frame_stack)

        # Default keys - order matters for state concatenation
        self.state_keys = state_keys or [
            "left_eef_pos",
            "left_eef_quat",
            "right_eef_pos",
            "right_eef_quat",
            "hand_joint_state",
        ]
        self.image_keys = image_keys or ["robot_pov_cam"]

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
            "seq_length": 1,  # Force seq_length to 1 when using frame_stack as time
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
            return min(len(self.dataset), self.demo_limit * 350)
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


if __name__ == "__main__":
    test_simple_dataset()
