"""Simple Sequence-based Dataset for Behavioral Cloning.

This module provides a light, self-contained sequence dataset for imitation
learning over Isaac Lab / robomimic-style HDF5 files. It supports:
- State-only, image-only, or state+image modalities
- Fixed-length sequences with optional front/back padding (boundary repeat)
- Global min-max action normalization (single scalar range)
- Optional multi-camera images via a simple nested structure

Focus is on clarity and minimalism — no training code or unused configs here.
"""

import os
import time
from typing import Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
try:  # optional, for progress bars during preprocessing
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

__all__ = ["SequenceDataset"]


class SequenceDataset(torch.utils.data.Dataset):
    """Simple sequence dataset for behavioral cloning.

    Sequencing (pad_sequence=True):
    - For a demo of length ``L`` and sequence length ``T``, sequences start at
      indices ``s ∈ [-(T-1), ..., L-1]`` (total ``L + T - 1`` sequences).
    - Each requested timestep ``(s + t)`` is clamped to ``[0, L-1]`` to repeat
      boundary frames for front/back padding.

    Example (L=3, T=5): start indices = [-4, -3, -2, -1, 0, 1, 2]
    - s=-4 → indices [-4,-3,-2,-1,0] → clamp → [0,0,0,0,0]
    - s=-1 → [-1,0,1,2,3] → clamp → [0,0,1,2,2]
    - s= 2 → [2,3,4,5,6] → clamp → [2,2,2,2,2]

    Outputs from ``__getitem__`` (float32 tensors):
    - actions: ``(A,)`` for the last timestep in the window; if ``normalize_actions=True`` then
      ``actions = (actions_raw - center) / scale`` where ``center`` and ``scale``
      are computed globally over all demos and action dims (single scalar range).
    - state_obs (if requested): ``(T, S)`` concatenation of the provided
      ``state_keys`` in file order; raises ``KeyError`` if a key is missing.
    - image_obs (if requested):
        - single camera → ``(T, C, H, W)``
        - multi-camera → dict ``{key: (T, C, H, W)}``
      Images are converted to CHW and normalized to [0,1] if integer dtype;
      then resized to the requested ``img_size=(C,H,W)``.
    """

    def __init__(
        self,
        hdf5_path: str,
        modality_type: str = "state+image",  # one of: "image", "state", "state+image"
        state_keys: list = None,
        image_keys: list = None,
        sequence_length: int = 1,
        pad_sequence: bool = True,
        normalize_actions: bool = True,
        demo_limit: int = None,
        img_size: Union[Sequence[int], Tuple[int, int, int]] = (3, 160, 256),
        # Preprocessing/cache options
        use_preprocessed: bool = True,
        preprocessed_path: str | None = None,
        concat_state_key: str = "state_concat",
        compression: str | None = None,
        force_rebuild: bool = False,
        preprocess_chunk_len: int = 1024,
        preprocess_device: str = "auto",
        in_memory: bool = False,
    ):
        """
        Simple sequence dataset initialization.

        Args:
            hdf5_path: Path to HDF5 dataset file
            modality_type: "image", "state", or "state+image"
            state_keys: State observation keys (None for defaults)
            image_keys: Image observation keys (None for defaults). If multiple
                keys are provided, images are returned as a dict mapping each
                key to a tensor of shape ``(T, C, H, W)``. For a single key,
                images are returned directly as a tensor of shape ``(T,C,H,W)``.
            sequence_length: Length of sequences (>=1)
            pad_sequence: Pad sequences at beginning/end
            normalize_actions: Normalize actions across all dimensions
            demo_limit: Max demos to load (None for all)
            img_size: Desired image size as (C, H, W). Images are converted to
                channel-first and resized if necessary. Integer dtypes are
                normalized to [0,1] by dividing by 255.
        """
        # Configuration
        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.modality_type = modality_type
        self.sequence_length = sequence_length
        self.pad_sequence = pad_sequence
        self.normalize_actions = normalize_actions
        self.img_size = tuple(img_size)  # (C, H, W)
        self.concat_state_key = concat_state_key
        self.use_preprocessed = use_preprocessed
        self.force_rebuild = force_rebuild
        self.compression = compression
        self.preprocess_chunk_len = int(preprocess_chunk_len)
        self.preprocess_device = str(preprocess_device)
        self.in_memory = bool(in_memory)
        self._is_preprocessed = False
        self._mem_data = None

        assert self.sequence_length >= 1, "sequence_length must be >= 1"
        assert self.modality_type in {"image", "state", "state+image"}

        # Default keys
        self.state_keys = state_keys or [
            "left_eef_pos",
            "hand_joint_state",
            "right_eef_pos",
            "right_eef_quat",
            "left_eef_quat",
        ]
        self.image_keys = image_keys or ["robot_pov_cam"]

        # Optionally preprocess the dataset into a fast-read HDF5
        self.src_hdf5_path = self.hdf5_path
        if self.use_preprocessed:
            self.hdf5_path = self._ensure_preprocessed(self.hdf5_path, preprocessed_path)
            try:
                with h5py.File(self.hdf5_path, "r") as fpp:
                    if "img_size" in fpp.attrs:
                        self._is_preprocessed = True
            except Exception:
                pass
            # Ensure we read concatenated state vector if present in preprocessed file
            if self.concat_state_key and self.modality_type in ("state", "state+image"):
                try:
                    with h5py.File(self.hdf5_path, "r") as fchk:
                        data_grp = fchk.get("data", None)
                        if data_grp is not None:
                            # Probe first demo for the concat key
                            demos = list(data_grp.keys())
                            if demos:
                                probe_path = f"data/{demos[0]}/obs/{self.concat_state_key}"
                                if probe_path in fchk:
                                    self.state_keys = [self.concat_state_key]
                except Exception:
                    # Best effort fallback: assume concat key exists in preprocessed files
                    self.state_keys = [self.concat_state_key]

        # Load demos and create sequence indices
        self._load_demos(demo_limit)

        # Action normalization stats (prefer file-level attrs if present)
        self.action_stats = None
        if self.normalize_actions:
            try:
                with h5py.File(self.hdf5_path, "r") as fchk:
                    amin = fchk.attrs.get("action_min", None)
                    amax = fchk.attrs.get("action_max", None)
                    if amin is not None and amax is not None:
                        self.action_stats = {"min": float(amin), "max": float(amax)}
            except Exception:
                self.action_stats = None
        if self.action_stats is None and self.normalize_actions:
            self.action_stats = self._compute_action_stats()

        # Optionally preload entire dataset into memory (fastest, high RAM)
        if self.in_memory:
            self._mem_data = {}
            with h5py.File(self.hdf5_path, "r") as f:
                for demo_id in self.demos:
                    g = f[f"data/{demo_id}"]
                    entry = {"actions": np.asarray(g["actions"][()])}
                    # States
                    if self.modality_type in ("state", "state+image"):
                        obs = g["obs"]
                        if self.concat_state_key and self.concat_state_key in obs:
                            entry["state"] = np.asarray(obs[self.concat_state_key][()])
                        else:
                            parts = []
                            for k in self.state_keys:
                                if k in obs:
                                    arr = np.asarray(obs[k][()])
                                    parts.append(arr.reshape(arr.shape[0], -1))
                            entry["state"] = (
                                np.concatenate(parts, axis=-1).astype(np.float32) if parts else None
                            )
                    # Images
                    if self.modality_type in ("image", "state+image"):
                        obs = g["obs"]
                        imgs = {}
                        for k in self.image_keys:
                            if k in obs:
                                imgs[k] = np.asarray(obs[k][()])
                        entry["images"] = imgs
                    self._mem_data[demo_id] = entry

        # Lazily opened HDF5 handle per worker (opened on first __getitem__)
        self._h5 = None

    def __del__(self):
        try:
            if hasattr(self, "_h5") and self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    def _load_demos(self, demo_limit):
        """Load demos and build start indices for each sequence window.

        When ``pad_sequence=True``, start indices include negative values and
        extend beyond the demo end, enabling boundary-repeat padding. When
        ``pad_sequence=False``, only fully-contained windows are emitted.
        """
        with h5py.File(self.hdf5_path, "r") as f:
            self.demos = sorted(f["data"].keys(), key=lambda x: int(x[5:]))
            if demo_limit:
                self.demos = self.demos[:demo_limit]

            self.sequences = []
            for demo_id in self.demos:
                demo_grp = f[f"data/{demo_id}"]
                # Prefer attribute if present, otherwise infer from actions length
                if "num_samples" in demo_grp.attrs:
                    demo_len = int(demo_grp.attrs["num_samples"])
                else:
                    demo_len = int(demo_grp["actions"].shape[0])
                if self.pad_sequence:
                    # Allow starts before and beyond the demo for boundary padding
                    start_range = demo_len + self.sequence_length - 1
                    for start in range(start_range):
                        # Negative values indicate front padding
                        actual_start = start - (self.sequence_length - 1)
                        self.sequences.append((demo_id, actual_start))
                else:
                    # Only sequences that fit entirely in-bounds
                    for start in range(demo_len - self.sequence_length + 1):
                        self.sequences.append((demo_id, start))

    def _compute_action_stats(self):
        """Compute global min/max for actions across all demos efficiently.

        Uses streaming per-demo updates to avoid allocating a single large
        concatenated array, which can be slow and memory-heavy on big datasets.
        """
        min_v = np.inf
        max_v = -np.inf
        with h5py.File(self.hdf5_path, "r") as f:
            for demo_id in self.demos:
                acts = f[f"data/{demo_id}/actions"]
                # HDF5 supports partial reads; read in moderately-sized chunks if large
                if acts.size > 0:
                    # Read in chunks along the first dimension when length is large
                    L = acts.shape[0]
                    step = max(1, min(4096, L))
                    for s in range(0, L, step):
                        e = min(L, s + step)
                        a = np.asarray(acts[s:e])
                        # Flatten last dims, compute min/max
                        min_v = min(min_v, float(np.min(a)))
                        max_v = max(max_v, float(np.max(a)))

        if not np.isfinite(min_v) or not np.isfinite(max_v):
            # Fallback in case of empty dataset
            min_v, max_v = 0.0, 1.0

        return {"min": float(min_v), "max": float(max_v)}

    def _ensure_preprocessed(self, input_path: str, output_path: str | None) -> str:
        """Create or reuse a preprocessed HDF5 for fast reads.

        - Images: (L,C,H,W) float32 in [0,1], resized to img_size
        - States: concatenated into a single key if concat_state_key is provided
        - Actions: copied as-is; file attrs store action_min/action_max
        """
        C, H, W = self.img_size
        if output_path is None or len(str(output_path)) == 0:
            stem, ext = os.path.splitext(input_path)
            output_path = f"{stem}_preproc_{C}x{H}x{W}.hdf5"

        # Decide whether to rebuild
        rebuild = self.force_rebuild or (not os.path.isfile(output_path))
        if not rebuild:
            try:
                with h5py.File(output_path, "r") as f:
                    img_size_attr = tuple(f.attrs.get("img_size", ()))
                    if img_size_attr != (C, H, W):
                        rebuild = True
            except Exception:
                rebuild = True
        if not rebuild:
            return output_path

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        action_min = np.inf
        action_max = -np.inf
        print(f"[SequenceDataset] Preprocessing HDF5 → {output_path}")
        t0 = time.time()
        with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
            data_dst = dst.create_group("data")
            demos = sorted(src["data"].keys(), key=lambda x: int(x[5:]))
            iterator = _tqdm(demos, desc="Preprocessing demos", leave=False) if _tqdm else demos
            for demo_id in iterator:
                g_src = src["data"][demo_id]
                g_dst = data_dst.create_group(demo_id)
                if "num_samples" in g_src.attrs:
                    g_dst.attrs["num_samples"] = int(g_src.attrs["num_samples"])
                L = int(g_src["actions"].shape[0])

                # Actions (chunked copy + streaming min/max)
                acts_src = g_src["actions"]
                act_shape = acts_src.shape
                act_chunks = (min(L, max(1, self.preprocess_chunk_len)),) + act_shape[1:]
                ds_actions = g_dst.create_dataset(
                    "actions",
                    shape=act_shape,
                    dtype=acts_src.dtype,
                    compression=self.compression,
                    chunks=act_chunks,
                )
                if acts_src.size > 0:
                    step = max(1, min(self.preprocess_chunk_len * 4, L))
                    for s in range(0, L, step):
                        e = min(L, s + step)
                        a = np.asarray(acts_src[s:e])
                        ds_actions[s:e] = a
                        action_min = min(action_min, float(np.min(a)))
                        action_max = max(action_max, float(np.max(a)))

                # Observations
                obs_src = g_src["obs"]
                obs_dst = g_dst.create_group("obs")

                # State concatenation if requested (chunked)
                if self.modality_type in ("state", "state+image") and self.concat_state_key:
                    # Compute total state dim
                    dims = []
                    keys_present = [k for k in (self.state_keys or []) if k in obs_src]
                    for k in keys_present:
                        shape_k = obs_src[k].shape
                        dims.append(int(np.prod(shape_k[1:])))
                    total_dim = int(sum(dims)) if dims else 1
                    ds_state = obs_dst.create_dataset(
                        self.concat_state_key,
                        shape=(L, total_dim),
                        dtype=np.float32,
                        compression=self.compression,
                        chunks=(min(L, max(1, self.preprocess_chunk_len)), total_dim),
                    )
                    if not dims:
                        ds_state[...] = 0.0
                    else:
                        step = max(1, self.preprocess_chunk_len)
                        for s in range(0, L, step):
                            e = min(L, s + step)
                            out = np.empty((e - s, total_dim), dtype=np.float32)
                            off = 0
                            for k, d in zip(keys_present, dims):
                                arr = np.asarray(obs_src[k][s:e]).reshape(e - s, -1)
                                out[:, off : off + d] = arr.astype(np.float32)
                                off += d
                            ds_state[s:e, :] = out
                elif self.modality_type in ("state", "state+image"):
                    for k in (self.state_keys or []):
                        if k in obs_src:
                            obs_dst.create_dataset(
                                k,
                                data=np.asarray(obs_src[k][()]),
                                compression=self.compression,
                            )

                # Images
                if self.modality_type in ("image", "state+image"):
                    # Resolve device selection (auto -> cuda if available)
                    dev_str = (self.preprocess_device or "auto").lower()
                    if dev_str == "auto":
                        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    else:
                        try:
                            torch_device = torch.device(dev_str)
                            if torch_device.type == "cuda" and not torch.cuda.is_available():
                                torch_device = torch.device("cpu")
                        except Exception:
                            torch_device = torch.device("cpu")
                    for k in (self.image_keys or []):
                        if k not in obs_src:
                            continue
                        # Precreate destination dataset with chunking
                        ds_img = obs_dst.create_dataset(
                            k,
                            shape=(L, C, H, W),
                            dtype=np.float32,
                            compression=self.compression,
                            chunks=(min(L, max(1, self.preprocess_chunk_len)), C, H, W),
                        )
                        step = max(1, self.preprocess_chunk_len)
                        for s in range(0, L, step):
                            e = min(L, s + step)
                            arr = np.asarray(obs_src[k][s:e])  # (N,H,W,C) or (N,C,H,W) or (N,H,W)
                            # To float32 in [0,1]
                            if np.issubdtype(arr.dtype, np.integer):
                                arr = arr.astype(np.float32) / 255.0
                            else:
                                arr = arr.astype(np.float32)
                            # To CHW
                            if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
                                arr = np.transpose(arr, (0, 3, 1, 2))
                            elif arr.ndim == 3:
                                arr = arr[:, None, :, :]
                            # Match channels
                            C_actual = arr.shape[1]
                            if C_actual != C:
                                if C_actual > C:
                                    arr = arr[:, :C]
                                else:
                                    reps = (C + C_actual - 1) // C_actual
                                    arr = np.tile(arr, (1, reps, 1, 1))[:, :C]
                            # Resize on chosen device
                            if (arr.shape[2], arr.shape[3]) != (H, W):
                                t = torch.from_numpy(arr).to(torch_device)
                                t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
                                arr = t.to("cpu").numpy()
                            ds_img[s:e, :, :, :] = arr

            # File-level attributes
            dst.attrs["img_size"] = np.array([C, H, W], dtype=np.int64)
            if np.isfinite(action_min) and np.isfinite(action_max):
                dst.attrs["action_min"] = float(action_min)
                dst.attrs["action_max"] = float(action_max)

        # If concatenated states are used, switch state_keys to the concat key
        if self.concat_state_key and self.modality_type in ("state", "state+image"):
            self.state_keys = [self.concat_state_key]
        dt = time.time() - t0
        print(f"[SequenceDataset] Preprocessing complete in {dt:.1f}s; wrote {output_path}")
        return output_path

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get sequence sample.

        Returns a dict with keys depending on ``modality_type``:
        - Always: ``{"actions": (A,)}``
        - If ``state`` in modality: adds ``{"state_obs": (T, S)}``
        - If ``image`` in modality: adds
            - single camera → ``{"image_obs": (T, C, H, W)}``
            - multi-camera → ``{"image_obs": {key: (T, C, H, W)}}``

        Action values are normalized if ``normalize_actions=True``.
        """
        demo_id, start_idx = self.sequences[idx]

        mem = self._mem_data
        if mem is None:
            f = self._get_h5()
            demo_grp = f[f"data/{demo_id}"]
        else:
            demo_grp = None
        demo_len = (
            int(demo_grp.attrs["num_samples"]) if (demo_grp is not None and "num_samples" in demo_grp.attrs)
            else int((demo_grp["actions"].shape[0] if demo_grp is not None else mem[demo_id]["actions"].shape[0]))
        )

        # Vectorized clamped indices for the window
        idxs = np.arange(start_idx, start_idx + self.sequence_length)
        if demo_len > 0:
            idxs = np.clip(idxs, 0, demo_len - 1)
        else:
            idxs = np.zeros(self.sequence_length, dtype=np.int64)

        result = {}

        # Action (A,) at the last timestep only
        if demo_grp is not None:
            act_ds = demo_grp["actions"]
            L_act = act_ds.shape[0]
        else:
            act_arr = mem[demo_id]["actions"]
            L_act = act_arr.shape[0]
        if L_act == 0:
            A = int(act_arr.shape[1]) if (demo_grp is None and len(act_arr.shape) > 1) else (int(act_ds.shape[1]) if (demo_grp is not None and len(act_ds.shape) > 1) else 1)
            actions = np.zeros((A,), dtype=np.float32)
        else:
            last_idx = int(idxs[-1])
            last_idx = max(0, min(last_idx, L_act - 1))
            actions = (
                np.asarray(act_ds[last_idx]) if demo_grp is not None else act_arr[last_idx]
            )
        if self.normalize_actions and self.action_stats:
            # Map to [0, 1] using single global min / max
            denom = (self.action_stats["max"] - self.action_stats["min"]) + 1e-8
            actions = (actions - self.action_stats["min"]) / denom
        result["actions"] = torch.from_numpy(actions.astype(np.float32))

        # States (T, S)
        if self.modality_type in ["state", "state+image"]:
            state_parts = []
            for key in self.state_keys:
                ds_path = f"data/{demo_id}/obs/{key}"
                if demo_grp is None:
                    # Expect concatenated state present
                    if key == self.concat_state_key and self._mem_data[demo_id].get("state") is not None:
                        arr = self._mem_data[demo_id]["state"][idxs]
                    else:
                        raise KeyError(
                            f"Missing state observation key '{key}' for demo {demo_id} in memory."
                        )
                else:
                    if ds_path not in f:
                        raise KeyError(
                            f"Missing state observation key '{key}' at '{ds_path}'."
                        )
                    arr = self._h5_gather(f[ds_path], idxs)
                state_parts.append(arr.reshape(arr.shape[0], -1))
            state_arr = (
                np.concatenate(state_parts, axis=-1)
                if state_parts
                else np.zeros((self.sequence_length, 1), dtype=np.float32)
            )
            result["state_obs"] = torch.from_numpy(state_arr.astype(np.float32))

        # Images (single camera or dict of cameras): (T, C, H, W)
        if self.modality_type in ["image", "state+image"]:
            C_desired, H_desired, W_desired = self.img_size
            if len(self.image_keys) == 1:
                key = self.image_keys[0]
                if demo_grp is None:
                    arr = self._mem_data[demo_id]["images"].get(key, None)
                    if arr is None:
                        img = torch.zeros((self.sequence_length, C_desired, H_desired, W_desired), dtype=torch.float32)
                    else:
                        img = torch.from_numpy(arr[idxs].astype(np.float32))
                else:
                    img = self._read_image_sequence(
                        f, demo_id, key, idxs, C_desired, H_desired, W_desired
                    )
                result["image_obs"] = img
            else:
                imgs = {}
                for key in self.image_keys:
                    if demo_grp is None:
                        arr = self._mem_data[demo_id]["images"].get(key, None)
                        if arr is None:
                            imgs[key] = torch.zeros((self.sequence_length, C_desired, H_desired, W_desired), dtype=torch.float32)
                        else:
                            imgs[key] = torch.from_numpy(arr[idxs].astype(np.float32))
                    else:
                        imgs[key] = self._read_image_sequence(
                            f, demo_id, key, idxs, C_desired, H_desired, W_desired
                        )
                result["image_obs"] = imgs

        return result

    def _get_h5(self):
        if self._h5 is None:
            # Open once per worker for speed
            self._h5 = h5py.File(self.hdf5_path, "r", libver="latest", swmr=True)
        return self._h5

    def _read_image_sequence(
        self, f, demo_id, key, idxs, C_desired, H_desired, W_desired
    ):
        path = f"data/{demo_id}/obs/{key}"
        if path not in f:
            # Return zeros if missing
            zeros = torch.zeros(
                (self.sequence_length, C_desired, H_desired, W_desired),
                dtype=torch.float32,
            )
            return zeros
        # If preprocessed, this is already (T,C,H,W) float32 in [0,1]
        arr = self._h5_gather(f[path], idxs)
        if self._is_preprocessed:
            return torch.from_numpy(np.asarray(arr))
        arr = np.asarray(arr, dtype=np.float32)
        # Defensive path in case file isn't preprocessed
        if arr.ndim != 4 or arr.shape[1] not in (1, 3, 4) or arr.shape[2:] != (H_desired, W_desired):
            # Normalize integers
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float32) / 255.0
            # To CHW
            if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
                arr = np.transpose(arr, (0, 3, 1, 2))
            elif arr.ndim == 3:
                arr = arr[:, None, :, :]
            # Match channels
            C_actual = arr.shape[1]
            if C_actual != C_desired:
                if C_actual > C_desired:
                    arr = arr[:, :C_desired]
                else:
                    reps = (C_desired + C_actual - 1) // C_actual
                    arr = np.tile(arr, (1, reps, 1, 1))[:, :C_desired]
            # Resize
            if arr.shape[2:] != (H_desired, W_desired):
                t = torch.from_numpy(arr)
                t = F.interpolate(t, size=(H_desired, W_desired), mode="bilinear", align_corners=False)
                arr = t.numpy()
        return torch.from_numpy(arr)

    @staticmethod
    def _h5_gather(ds, idxs: np.ndarray):
        """Gather rows from an HDF5 dataset using sorted-unique selection.

        HDF5 fancy indexing requires indices to be in increasing order and may
        fail on repeated values. This helper performs a unique, sorted read and
        reconstructs the original ordering (including repeats) via inverse map.
        """
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.ndim != 1:
            idxs = idxs.reshape(-1)
        unique, inverse = np.unique(idxs, return_inverse=True)
        vals_unique = ds[unique]
        return np.asarray(vals_unique)[inverse]

    def _get_state_obs(self, f, demo_id, timestep):
        """Extract and concatenate state observations for known keys.

        Raises a KeyError if any requested key is not present in the file.
        """
        state_parts = []
        for key in self.state_keys:
            dataset_path = f"data/{demo_id}/obs/{key}"
            if dataset_path not in f:
                raise KeyError(
                    f"Missing state observation key '{key}' at '{dataset_path}'."
                )
            data = f[dataset_path][timestep]
            state_parts.append(np.asarray(data).reshape(-1))
        return (
            np.concatenate(state_parts)
            if state_parts
            else np.array([0.0], dtype=np.float32)
        )

    def _get_image_obs(self, f, demo_id, timestep, key):
        """Extract an image as float32 CHW in [0, 1], resized to ``self.img_size``.

        - If the dataset is uint8 / integer, values are divided by 255.
        - Supports HWC or CHW; grayscale is expanded to the desired channels
          by replication or slicing to match ``img_size[0]``.
        - If the key is missing, returns zeros of shape ``(C,H,W)``.
        """
        path = f"data/{demo_id}/obs/{key}"
        C_desired, H_desired, W_desired = self.img_size

        if path in f:
            img = np.asarray(f[path][timestep])  # could be HWC, CHW, or HW
            # Normalize by dtype
            if np.issubdtype(img.dtype, np.integer):
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)

            # To CHW
            if img.ndim == 2:  # HW → 1xHxW
                img = img[None, ...]
            elif img.ndim == 3:
                if img.shape[0] in (1, 3, 4):  # CHW
                    pass
                elif img.shape[-1] in (1, 3, 4):  # HWC → CHW
                    img = np.transpose(img, (2, 0, 1))
                else:
                    raise ValueError(
                        f"Unrecognized image shape for key '{key}': {img.shape}"
                    )
            else:
                raise ValueError(f"Unsupported image ndim for key '{key}': {img.ndim}")

            # Match channel count (slice or replicate)
            C_actual = img.shape[0]
            if C_actual != C_desired:
                if C_actual > C_desired:
                    img = img[:C_desired, ...]
                else:
                    reps = (C_desired + C_actual - 1) // C_actual
                    img = np.tile(img, (reps, 1, 1))[:C_desired]

            # Resize if needed using torch interpolate
            H_actual, W_actual = img.shape[1], img.shape[2]
            if (H_actual, W_actual) != (H_desired, W_desired):
                t = torch.from_numpy(img[None, ...])  # 1xCxHxW
                t = F.interpolate(
                    t, size=(H_desired, W_desired), mode="bilinear", align_corners=False
                )
                img = t.squeeze(0).numpy()
            return img

        # Missing key → zeros
        return np.zeros((C_desired, H_desired, W_desired), dtype=np.float32)

    def _get_actions(self, f, demo_id, timestep):
        """Extract and normalize actions."""
        actions = f[f"data/{demo_id}/actions"][timestep]

        # Normalize if enabled
        if self.normalize_actions and self.action_stats:
            actions = (actions - self.action_stats["min"]) / (
                self.action_stats["max"] - self.action_stats["min"] + 1e-8
            )

        return actions

    def denormalize_actions(self, normalized_actions):
        """Denormalize actions back to original range."""
        if not self.normalize_actions or not self.action_stats:
            return normalized_actions

        denom = (self.action_stats["max"] - self.action_stats["min"]) + 1e-8
        return normalized_actions * denom + self.action_stats["min"]


def test_sequence_padding():
    """Minimal tests with clear ✓/✗ reporting.

    Verifies:
    - Sequencing start indices and boundary padding behavior
    - Image shape (T,C,H,W) and normalization to [0,1]
    - No-padding exact-fit window selection
    - Fallback to infer demo length from actions when num_samples is absent
    - Strict KeyError on missing state key
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp_file:
        test_file = tmp_file.name

    try:
        # Build a tiny dataset with one short and one long demo
        with h5py.File(test_file, "w") as f:
            data = f.create_group("data")

            # Short demo (L=3)
            g0 = data.create_group("demo_0")
            g0.attrs["num_samples"] = 3
            obs0 = g0.create_group("obs")
            obs0.create_dataset(
                "robot_joint_pos",
                data=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
            )
            # uint8 image with small values → should normalize to [0,1]
            img0 = np.zeros((3, 64, 64, 3), dtype=np.uint8)
            for t in range(3):
                img0[t, 0, 0, 0] = t  # embed a marker
            obs0.create_dataset("robot_pov_cam", data=img0)
            g0.create_dataset(
                "actions", data=np.array([[10.0], [11.0], [12.0]], dtype=np.float32)
            )

            # Long demo (L=8)
            g1 = data.create_group("demo_1")
            # Purposely omit num_samples to test fallback to actions length
            obs1 = g1.create_group("obs")
            obs1.create_dataset(
                "robot_joint_pos",
                data=np.array([[float(i)] for i in range(8)], dtype=np.float32),
            )
            img1 = np.zeros((8, 64, 64, 3), dtype=np.uint8)
            for t in range(8):
                img1[t, 0, 0, 0] = t
            obs1.create_dataset("robot_pov_cam", data=img1)
            g1.create_dataset(
                "actions",
                data=np.array([[100.0 + i] for i in range(8)], dtype=np.float32),
            )

        # Short demo with padding (T=5)
        ds = SequenceDataset(
            hdf5_path=test_file,
            modality_type="state+image",
            state_keys=["robot_joint_pos"],
            image_keys=["robot_pov_cam"],
            sequence_length=5,
            pad_sequence=True,
            normalize_actions=False,
            demo_limit=1,
            img_size=(3, 64, 64),
        )
        results = []
        check = lambda ok: "✅" if ok else "❌"

        # Sequence starts for L=3, T=5
        starts = [s for (d, s) in ds.sequences if d == "demo_0"]
        expected_starts = list(range(-4, 3))
        ok_starts = starts == expected_starts
        print(f"{check(ok_starts)} start indices for L=3,T=5 are {expected_starts}")
        results.append(ok_starts)

        # Check specific sequences
        s_front = ds[0]["state_obs"].squeeze().tolist()  # start=-4 → all zeros
        s_mixed = ds[3]["state_obs"].squeeze().tolist()  # start=-1 → [0,0,1,2,2]
        s_nofr = ds[4]["state_obs"].squeeze().tolist()  # start=0  → [0,1,2,2,2]
        s_back = ds[6]["state_obs"].squeeze().tolist()  # start=2  → all twos

        ok_front = all(x == 0.0 for x in s_front)
        print(f"{check(ok_front)} front padding at start=-4 → all zeros")
        results.append(ok_front)

        ok_mixed = s_mixed == [0.0, 0.0, 1.0, 2.0, 2.0]
        print(f"{check(ok_mixed)} mixed padding at start=-1 → [0,0,1,2,2]")
        results.append(ok_mixed)

        ok_nofr = s_nofr[:3] == [0.0, 1.0, 2.0] and s_nofr[3:] == [2.0, 2.0]
        print(f"{check(ok_nofr)} no front padding at start=0 → [0,1,2,2,2]")
        results.append(ok_nofr)

        ok_back = all(x == 2.0 for x in s_back)
        print(f"{check(ok_back)} back padding at start=2 → all twos")
        results.append(ok_back)

        # Image checks: CHW and normalized to [0,1]
        img = ds[0]["image_obs"]  # (T,C,H,W)
        ok_img_shape = img.ndim == 4 and img.shape[1:] == (3, 64, 64)
        ok_img_norm = float(img.max()) <= 1.0 and float(img.min()) >= 0.0
        ok_img = ok_img_shape and ok_img_norm
        print(f"{check(ok_img)} image tensor is (T,3,64,64) and in [0,1]")
        results.append(ok_img)

        # No padding mode where T=L=3 → exactly one sequence
        ds_nopad = SequenceDataset(
            hdf5_path=test_file,
            modality_type="state",
            state_keys=["robot_joint_pos"],
            sequence_length=3,
            pad_sequence=False,
            normalize_actions=False,
            demo_limit=1,
        )
        ok_nopad = len(ds_nopad.sequences) == 1
        print(f"{check(ok_nopad)} no-padding exact fit yields 1 sequence")
        results.append(ok_nopad)

        # Fallback length for long demo (num_samples missing)
        ds_full = SequenceDataset(
            hdf5_path=test_file,
            modality_type="state",
            state_keys=["robot_joint_pos"],
            sequence_length=5,
            pad_sequence=True,
            normalize_actions=False,
            demo_limit=None,
        )
        starts1 = [s for (d, s) in ds_full.sequences if d == "demo_1"]
        ok_fallback = starts1 == list(range(-4, 8))
        print(f"{check(ok_fallback)} fallback length for demo_1 (L=8) → starts [-4..7]")
        results.append(ok_fallback)

        # Missing state key should raise KeyError
        try:
            ds_bad = SequenceDataset(
                hdf5_path=test_file,
                modality_type="state",
                state_keys=["missing_key"],
                sequence_length=3,
                pad_sequence=True,
                normalize_actions=False,
                demo_limit=1,
            )
            _ = ds_bad[0]  # trigger read
            ok_missing = False
        except KeyError:
            ok_missing = True
        print(f"{check(ok_missing)} raises KeyError on missing state key")
        results.append(ok_missing)

        all_ok = all(results)
        passed = sum(1 for r in results if r)
        total = len(results)
        print(f"Summary: {passed}/{total} tests passed.")
        return all_ok

    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == "__main__":
    # Run padding test
    test_sequence_padding()
