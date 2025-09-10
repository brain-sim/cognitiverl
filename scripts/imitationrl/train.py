"""Train Sequence Flow-BC with a minimal Trainer (Lightning-style simplicity).

Keeps the original functionality: preprocessed dataset fast reads, W&B (online
or offline) with artifact + code logging, checkpoints under the run folder,
image-sequence grids (train/val) saved locally and logged, and debugging
metrics (vs random/zero baselines).
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

try:
    from torchvision.utils import make_grid as tv_make_grid
except Exception:  # pragma: no cover
    tv_make_grid = None
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

# Try to use robomimic for datasets; fallback to local SequenceDataset
from robomimic.utils import obs_utils as RMObsUtils  # type: ignore
from robomimic.utils.dataset import SequenceDataset as RMSequenceDataset  # type: ignore

from scripts.imitationrl.model import SeqFlowPolicy, prepare_batch
from scripts.utils import load_args, print_dict, seed_everything

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


WANDB_DIR_DEFAULT = "/home/chandramouli/cognitiverl"


@dataclass
class Args:
    # Experiment
    device: str = "cuda:0"
    seed: int = 1
    exp_name: str = "seq_flow_bc"

    # Dataset
    dataset: str = "datasets/generated_dataset_gr1_nut_pouring.hdf5"
    use_robomimic: bool = True
    modality_type: str = "state+image"  # "state", "image", "state+image"
    state_keys: tuple = (
        "left_eef_pos",
        "hand_joint_state",
        "right_eef_pos",
        "right_eef_quat",
        "left_eef_quat",
    )
    image_keys: tuple = ("robot_pov_cam",)
    sequence_length: int = 10
    pad_sequence: bool = True
    normalize_actions: bool = True
    img_size: tuple = (3, 160, 256)
    train_split: float = 0.8
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    in_memory: bool = False

    # Model
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    ff_hidden: int = 512
    dropout: float = 0.1
    val_flow_steps: int = 32

    # Optimization
    batch_size: int = 512
    learning_rate: float = 1e-3
    num_epochs: int = 100
    eval_freq: int = 1
    ckpt_dir: str = "checkpoints_seq_flow_bc"
    # LR scheduler (MultiStep)
    use_scheduler: bool = True
    lr_milestones: tuple = (10, 20, 30, 60)
    lr_gamma: float = 0.5

    # Logging
    log: bool = True
    wandb_project: str = "bc"
    run_name: str = "seq_flow_bc"
    wandb_dir: str = WANDB_DIR_DEFAULT
    save_code: bool = True
    watch_model: bool = False
    # Image grid logging/saving
    log_image_grids: bool = True
    save_image_grids: bool = True
    image_grid_dir: str = None
    # AMP
    amp: bool = True
    amp_dtype: str = "bf16"


class Trainer:
    def __init__(self, args: Args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        seed_everything(args.seed, use_torch=True, torch_deterministic=True)

        # Dataset & loaders
        # Always use robomimic dataset
        if not args.use_robomimic:
            raise RuntimeError("Please set use_robomimic=True (required).")
        self.ds = self._build_rm_dataset()

        # Infer dims via a sample
        sample0 = self.ds[0]
        self.action_dim = int(sample0["actions"].shape[-1])
        self.state_dim = (
            int(sample0["state_obs"].shape[-1]) if "state_obs" in sample0 else 0
        )

        # Dataset info
        self._print_dataset_info(sample0)

        # Split
        train_len = int(args.train_split * len(self.ds))
        val_len = len(self.ds) - train_len
        train_set, val_set = random_split(self.ds, [train_len, val_len])
        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=(args.persistent_workers and args.num_workers > 0),
            prefetch_factor=args.prefetch_factor,
        )
        self.train_loader = DataLoader(
            train_set, shuffle=True, drop_last=True, **loader_kwargs
        )
        self.val_loader = DataLoader(
            val_set, shuffle=False, drop_last=False, **loader_kwargs
        )

        # Model & optimizer
        self.model = SeqFlowPolicy(self.state_dim, self.action_dim, args).to(
            self.device
        )
        # Faster optimizer variants if available
        opt_kwargs = {"lr": args.learning_rate}
        try:
            self.optim = torch.optim.AdamW(
                self.model.parameters(), fused=True, **opt_kwargs
            )  # type: ignore
        except Exception:
            try:
                self.optim = torch.optim.AdamW(
                    self.model.parameters(), foreach=True, **opt_kwargs
                )  # type: ignore
            except Exception:
                self.optim = torch.optim.AdamW(self.model.parameters(), **opt_kwargs)

        # LR Scheduler (MultiStep)
        self.scheduler = None
        if getattr(self.args, "use_scheduler", True):
            milestones = list(getattr(self.args, "lr_milestones", []))
            gamma = float(getattr(self.args, "lr_gamma", 1.0))
            if len(milestones) > 0 and gamma != 1.0:
                try:
                    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        self.optim, milestones=milestones, gamma=gamma
                    )
                except Exception:
                    self.scheduler = None

        # AMP scaler and dtype
        self.amp = (self.device.type == "cuda") and self.args.amp
        if self.amp:
            want_bf16 = self.args.amp_dtype.lower() == "bf16"
            bf16_ok = False
            try:
                bf16_ok = bool(
                    getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                )
            except Exception:
                bf16_ok = False
            self.amp_dtype = (
                torch.bfloat16 if (want_bf16 and bf16_ok) else torch.float16
            )
        else:
            self.amp_dtype = torch.float32
        self.scaler = torch.amp.GradScaler(
            enabled=self.amp and self.amp_dtype in (torch.bfloat16, torch.float16)
        )

        # W&B
        self.run = self._init_wandb()
        if self.run is not None and self.ds.action_stats is not None:
            wandb.log(
                {
                    "data/action_min": float(self.ds.action_stats["min"]),
                    "data/action_max": float(self.ds.action_stats["max"]),
                },
                step=0,
            )

        # Checkpoint directory under run folder
        if self.run is not None:
            ckpt_base = self.run.dir
            files_dir = (
                ckpt_base
                if os.path.basename(ckpt_base) == "files"
                else os.path.join(ckpt_base, "files")
            )
            self.args.ckpt_dir = os.path.join(files_dir, "checkpoints")
        os.makedirs(self.args.ckpt_dir, exist_ok=True)

        # Grid save dir
        self.grid_dir = None
        if args.save_image_grids:
            if self.run is not None:
                files_dir = (
                    os.path.join(self.run.dir, "files")
                    if os.path.basename(self.run.dir) != "files"
                    else self.run.dir
                )
                self.grid_dir = args.image_grid_dir or os.path.join(files_dir, "grids")
            else:
                base = os.path.dirname(self.args.ckpt_dir)
                self.grid_dir = args.image_grid_dir or os.path.join(base, "grids")
            os.makedirs(self.grid_dir, exist_ok=True)

        self.best_val = float("inf")
        self.global_step = 0

    # ---- robomimic dataset wrapper ----
    @staticmethod
    def _compute_action_stats_stream(hdf5_path: str) -> Dict[str, float]:
        import h5py
        import numpy as np

        min_v, max_v = np.inf, -np.inf
        with h5py.File(hdf5_path, "r") as f:
            for demo_id in sorted(f["data"].keys(), key=lambda x: int(x[5:])):
                acts = f[f"data/{demo_id}/actions"]
                L = acts.shape[0]
                step = max(1, min(4096, L))
                for s in range(0, L, step):
                    a = np.asarray(acts[s : s + step])
                    if a.size == 0:
                        continue
                    min_v = min(min_v, float(np.min(a)))
                    max_v = max(max_v, float(np.max(a)))
        if not np.isfinite(min_v) or not np.isfinite(max_v):
            min_v, max_v = 0.0, 1.0
        return {"min": float(min_v), "max": float(max_v)}

    def _build_rm_dataset(self):  # noqa: C901
        # Initialize robomimic ObsUtils with a minimal modality spec
        spec = {
            "obs": {
                "low_dim": list(self.args.state_keys),
                "rgb": list(self.args.image_keys),
            }
        }
        RMObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=spec)

        # Build a robomimic SequenceDataset and wrap to our expected tensors
        rm_ds = RMSequenceDataset(
            hdf5_path=self.args.dataset,
            obs_keys=tuple(list(self.args.state_keys) + list(self.args.image_keys)),
            dataset_keys=("actions",),
            frame_stack=1,
            seq_length=self.args.sequence_length,
            pad_frame_stack=True,
            pad_seq_length=bool(self.args.pad_sequence),
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode="low_dim",  # cache low-dim in RAM for speed
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
            load_next_obs=False,
        )

        action_stats = None
        if self.args.normalize_actions:
            try:
                # use file-level attrs if present (robomimic doesn't always set these)
                import h5py

                with h5py.File(self.args.dataset, "r") as f:
                    amin = f.attrs.get("action_min", None)
                    amax = f.attrs.get("action_max", None)
                    if amin is not None and amax is not None:
                        action_stats = {"min": float(amin), "max": float(amax)}
            except Exception:
                action_stats = None
            if action_stats is None:
                action_stats = self._compute_action_stats_stream(self.args.dataset)

        class RMWrapper(torch.utils.data.Dataset):
            def __init__(self, base, args, action_stats):
                self.base = base
                self.args = args
                self.action_stats = action_stats

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                sample = self.base[idx]
                # actions: (T,A) -> take last step
                actions = sample["actions"]
                if isinstance(actions, torch.Tensor):
                    actions_np = actions.detach().cpu().numpy()
                else:
                    actions_np = np.asarray(actions)
                if actions_np.ndim == 2:
                    act = actions_np[-1]
                else:
                    act = actions_np
                if self.args.normalize_actions and self.action_stats is not None:
                    denom = (self.action_stats["max"] - self.action_stats["min"]) + 1e-8
                    act = (act - self.action_stats["min"]) / denom
                out = {"actions": torch.from_numpy(act.astype(np.float32))}

                # Build state sequence (T,S) if state keys exist
                state_seq = []
                for k in self.args.state_keys:
                    if "obs" in sample and k in sample["obs"]:
                        arr = sample["obs"][k]
                    elif f"obs/{k}" in sample:
                        arr = sample[f"obs/{k}"]
                    else:
                        arr = None
                    if arr is None:
                        continue
                    arr_np = (
                        arr.detach().cpu().numpy()
                        if isinstance(arr, torch.Tensor)
                        else np.asarray(arr)
                    )
                    arr_np = (
                        arr_np.reshape(arr_np.shape[0], -1)
                        if arr_np.ndim > 1
                        else arr_np.reshape(-1, 1)
                    )
                    state_seq.append(arr_np)
                if len(state_seq) > 0:
                    state_np = np.concatenate(state_seq, axis=-1)
                    out["state_obs"] = torch.from_numpy(state_np.astype(np.float32))

                # Build image sequence (T,C,H,W) for first image key or dict for multiple
                if len(self.args.image_keys) == 1:
                    k = self.args.image_keys[0]
                    arr = None
                    if "obs" in sample and k in sample["obs"]:
                        arr = sample["obs"][k]
                    elif f"obs/{k}" in sample:
                        arr = sample[f"obs/{k}"]
                    if arr is not None:
                        img_np = (
                            arr.detach().cpu().numpy()
                            if isinstance(arr, torch.Tensor)
                            else np.asarray(arr)
                        )
                        # To (T,C,H,W) and float32 [0,1]
                        if img_np.ndim == 4 and img_np.shape[-1] in (1, 3, 4):
                            img_np = np.transpose(img_np, (0, 3, 1, 2))
                        if np.issubdtype(img_np.dtype, np.integer):
                            img_np = img_np.astype(np.float32) / 255.0
                        else:
                            img_np = img_np.astype(np.float32)
                        out["image_obs"] = torch.from_numpy(img_np)
                else:
                    imgs = {}
                    for k in self.args.image_keys:
                        arr = None
                        if "obs" in sample and k in sample["obs"]:
                            arr = sample["obs"][k]
                        elif f"obs/{k}" in sample:
                            arr = sample[f"obs/{k}"]
                        if arr is None:
                            continue
                        img_np = (
                            arr.detach().cpu().numpy()
                            if isinstance(arr, torch.Tensor)
                            else np.asarray(arr)
                        )
                        if img_np.ndim == 4 and img_np.shape[-1] in (1, 3, 4):
                            img_np = np.transpose(img_np, (0, 3, 1, 2))
                        if np.issubdtype(img_np.dtype, np.integer):
                            img_np = img_np.astype(np.float32) / 255.0
                        else:
                            img_np = img_np.astype(np.float32)
                        imgs[k] = torch.from_numpy(img_np)
                    if len(imgs) > 0:
                        out["image_obs"] = imgs
                return out

        wrapped = RMWrapper(rm_ds, self.args, action_stats)
        # attach stats for logging
        wrapped.action_stats = action_stats
        # mimic local dataset interface minimally
        self.ds = wrapped
        return wrapped

    def _init_wandb(self):
        if wandb is None:
            return None
        os.makedirs(self.args.wandb_dir, exist_ok=True)
        mode = "online" if self.args.log else "offline"
        os.environ["WANDB_MODE"] = mode
        os.environ["WANDB_DIR"] = self.args.wandb_dir
        run = wandb.init(
            project=self.args.wandb_project,
            name=self.args.run_name,
            config=vars(self.args),
            dir=self.args.wandb_dir,
            settings=wandb.Settings(code_dir=os.getcwd()),
            save_code=self.args.save_code,
        )
        try:
            wandb.run.log_code(root=os.getcwd())
        except Exception:
            pass
        if self.args.watch_model:
            wandb.watch(self.model, log="gradients", log_freq=200)
        return run

    def _print_dataset_info(self, sample0: Dict[str, torch.Tensor]):
        # Prefer robomimic dataset attributes if present
        base = getattr(self.ds, "base", None)
        num_demos = getattr(base, "n_demos", None)
        info = {
            "path": self.args.dataset,
            "modality": self.args.modality_type,
            "num_demos": int(num_demos) if num_demos is not None else None,
            "total_sequences": len(self.ds),
            "sequence_length": self.args.sequence_length,
            "pad_sequence": bool(self.args.pad_sequence),
        }
        a_shape = tuple(sample0["actions"].shape)
        info["action"] = {
            "shape": a_shape,
            "per_step_dim": a_shape[-1],
            "sequence_axis": None,
            "normalized": bool(self.args.normalize_actions),
        }
        if getattr(self.ds, "action_stats", None) is not None:
            info["action"].update(
                {
                    "min": float(self.ds.action_stats["min"]),
                    "max": float(self.ds.action_stats["max"]),
                }
            )
        if "state_obs" in sample0:
            s_shape = tuple(sample0["state_obs"].shape)
            info["state"] = {
                "shape": s_shape,
                "per_step_dim": s_shape[-1],
                "sequence_axis": s_shape[0],
            }
        else:
            info["state"] = None
        if "image_obs" in sample0:
            img = sample0["image_obs"]
            if isinstance(img, dict):
                imgs = {}
                for k, v in img.items():
                    shape = tuple(v.shape)
                    per_step_flat = (
                        int(np.prod(shape[-3:]))
                        if len(shape) == 4
                        else int(np.prod(shape[1:]))
                    )
                    imgs[k] = {
                        "shape": shape,
                        "sequence_axis": shape[0] if len(shape) >= 2 else None,
                        "per_step_CHW": shape[-3:] if len(shape) >= 3 else None,
                        "per_step_flat_dim": per_step_flat,
                    }
                info["images"] = imgs
            else:
                shape = tuple(img.shape)
                per_step_flat = (
                    int(np.prod(shape[-3:]))
                    if len(shape) == 4
                    else int(np.prod(shape[1:]))
                )
                info["images"] = {
                    "shape": shape,
                    "sequence_axis": shape[0] if len(shape) >= 2 else None,
                    "per_step_CHW": shape[-3:] if len(shape) >= 3 else None,
                    "per_step_flat_dim": per_step_flat,
                }
        else:
            info["images"] = None
        print("Dataset Info:")
        print_dict(info, nesting=4, color="cyan", attrs=["bold"])

    @staticmethod
    def _make_grid(seq: torch.Tensor) -> np.ndarray:
        # seq: (T, C, H, W) -> HxWxC uint8
        if tv_make_grid is not None:
            grid = tv_make_grid(seq, nrow=seq.shape[0], padding=2)
            img = grid.permute(1, 2, 0).cpu().numpy()
        else:
            img = (
                torch.cat([seq[t] for t in range(seq.shape[0])], dim=-1)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return img

    def _baseline_vectors(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.args.normalize_actions:
            rand = torch.rand_like(actions)
            zero = torch.full_like(actions, 0.5)
        else:
            if getattr(self.ds, "action_stats", None) is not None:
                a_min = float(self.ds.action_stats["min"])
                a_max = float(self.ds.action_stats["max"])
                rand = a_min + (a_max - a_min) * torch.rand_like(actions)
                zero = torch.full_like(actions, (a_min + a_max) * 0.5)
            else:
                rand = torch.randn_like(actions)
                zero = torch.zeros_like(actions)
        return rand, zero

    def _compute_metrics(
        self, pred: torch.Tensor, actions: torch.Tensor
    ) -> Dict[str, float]:
        mse = F.mse_loss(pred, actions, reduction="none").sum(dim=-1).mean()
        l2 = torch.norm(pred - actions, p=2, dim=-1).mean()
        cos = F.cosine_similarity(pred, actions, dim=-1).mean()
        rand, zero = self._baseline_vectors(actions)
        rnd_mse = F.mse_loss(rand, actions, reduction="none").sum(dim=-1).mean()
        rnd_l2 = torch.norm(rand - actions, p=2, dim=-1).mean()
        rnd_cos = F.cosine_similarity(rand, actions, dim=-1).mean()
        zero_mse = F.mse_loss(zero, actions, reduction="none").sum(dim=-1).mean()
        zero_l2 = torch.norm(zero - actions, p=2, dim=-1).mean()
        zero_cos = F.cosine_similarity(zero, actions, dim=-1).mean()
        imp_mse = (
            ((rnd_mse - mse) / (rnd_mse + 1e-8)) * 100.0
            if rnd_mse > 0
            else torch.tensor(0.0)
        )
        return {
            "mse": float(mse.item()),
            "l2": float(l2.item()),
            "cosine": float(cos.item()),
            "random_mse": float(rnd_mse.item()),
            "random_l2": float(rnd_l2.item()),
            "random_cosine": float(rnd_cos.item()),
            "zero_mse": float(zero_mse.item()),
            "zero_l2": float(zero_l2.item()),
            "zero_cosine": float(zero_cos.item()),
            "improve_mse_pct": float(imp_mse.item()),
            "better_than_random": float(1.0 if mse < rnd_mse else 0.0),
        }

    def _log_grid(
        self, tag: str, seq: torch.Tensor, metrics: Dict[str, float], epoch: int
    ):
        img = self._make_grid(seq)
        # Save locally
        if self.grid_dir is not None and imageio is not None:
            out_path = os.path.join(self.grid_dir, f"{tag}_epoch_{epoch:04d}.png")
            try:
                imageio.imwrite(out_path, img)
            except Exception:
                pass
        # Log to W&B
        if self.run is not None and self.args.log_image_grids:
            log = {f"{tag}/image_grid": wandb.Image(img)}
            # Attach metrics with the same tag
            for k, v in metrics.items():
                log[f"{tag}/{k}"] = v
            wandb.log(log, step=self.global_step)

    @staticmethod
    def _first_sample(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if "actions" in batch:
            out["actions"] = batch["actions"][:1].contiguous()
        if "state_obs" in batch:
            out["state_obs"] = batch["state_obs"][:1].contiguous()
        if "image_obs" in batch:
            img = batch["image_obs"]
            if isinstance(img, dict):
                k = sorted(img.keys())[0]
                out["image_obs"] = {k: img[k][:1].contiguous()}
            else:
                out["image_obs"] = img[:1].contiguous()
        return out

    def train_one_epoch(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, torch.Tensor]]]:
        self.model.train()
        losses = []
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.args.num_epochs} [train]",
            leave=False,
        )
        preview_batch = None
        for batch in pbar:
            t0 = time.time()
            state_seq, image_seq, actions = prepare_batch(
                batch, self.device, self.args.modality_type
            )
            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.amp,
                dtype=getattr(self, "amp_dtype", torch.float16),
            ):
                loss = self.model.compute_flow_loss(state_seq, image_seq, actions)
            self.optim.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            # Unscale before clipping for AMP
            self.scaler.unscale_(self.optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            ).item()
            self.scaler.step(self.optim)
            self.scaler.update()
            losses.append(loss.item())
            self.global_step += 1
            lr = self.optim.param_groups[0]["lr"]
            ips = actions.size(0) / max(1e-6, (time.time() - t0))
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{lr:.2e}",
                gn=f"{grad_norm:.2f}",
                ips=f"{ips:.1f}",
            )
            if self.run is not None:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/grad_norm": float(grad_norm),
                        "lr": float(lr),
                        "epoch": epoch,
                    },
                    step=self.global_step,
                )
            # Optional: per-step train metrics (no image grids)
            if getattr(self.args, "per_step_metrics", False) and self.run is not None:
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type="cuda",
                        enabled=self.amp,
                        dtype=getattr(self, "amp_dtype", torch.float16),
                    ):
                        pred_step = self.model.sample_actions(
                            state_seq,
                            image_seq,
                            steps=int(getattr(self.args, "per_step_metrics_steps", 4)),
                        )
                    m = self._compute_metrics(pred_step, actions)
                wandb.log(
                    {f"train_step/{k}": v for k, v in m.items()}, step=self.global_step
                )
            if preview_batch is None:
                preview_batch = self._first_sample(batch)
        return float(np.mean(losses)) if losses else 0.0, preview_batch

    @torch.no_grad()
    def validate(
        self, epoch: int
    ) -> Tuple[Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
        self.model.eval()
        val_mse, val_l2, val_cos = [], [], []
        vbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.args.num_epochs} [val]",
            leave=False,
        )
        for batch in vbar:
            state_seq, image_seq, actions = prepare_batch(
                batch, self.device, self.args.modality_type
            )
            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.amp,
                dtype=getattr(self, "amp_dtype", torch.float16),
            ):
                pred = self.model.sample_actions(
                    state_seq, image_seq, steps=self.args.val_flow_steps
                )
            mse_b = (
                F.mse_loss(pred, actions, reduction="none").sum(dim=-1).mean().item()
            )
            l2_b = torch.norm(pred - actions, p=2, dim=-1).mean().item()
            cos_b = F.cosine_similarity(pred, actions, dim=-1).mean().item()
            val_mse.append(mse_b)
            val_l2.append(l2_b)
            val_cos.append(cos_b)
            vbar.set_postfix(val_mse=f"{mse_b:.4f}", val_l2=f"{l2_b:.4f}")
            # Optional: per-step val metrics (no image grids)
            if getattr(self.args, "per_step_metrics", False) and self.run is not None:
                m_step = self._compute_metrics(pred, actions)
                wandb.log(
                    {f"val_step/{k}": v for k, v in m_step.items()},
                    step=self.global_step,
                )
        out = {
            "mse": float(np.mean(val_mse)) if val_mse else 0.0,
            "l2": float(np.mean(val_l2)) if val_l2 else 0.0,
            "cosine": float(np.mean(val_cos)) if val_cos else 0.0,
        }
        # Provide a preview batch for grid logging
        try:
            preview = self._first_sample(next(iter(self.val_loader)))
        except Exception:
            preview = None
        return out, preview

    def save_checkpoint(self, path: str, epoch: int, best_val: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val,
                "args": vars(self.args),
                "action_stats": self.ds.action_stats
                if self.args.normalize_actions
                else None,
            },
            path,
        )

    def fit(self):
        for epoch in range(1, self.args.num_epochs + 1):
            epoch_start = time.time()
            train_loss, train_preview = self.train_one_epoch(epoch)

            val_metrics = None
            val_preview = None
            if epoch % self.args.eval_freq == 0:
                # Aggregate validation metrics
                val_metrics, val_preview = self.validate(epoch)

                # Full metrics vs baselines using a preview batch
                if val_preview is not None and self.args.modality_type in (
                    "image",
                    "state+image",
                ):
                    img_seq = val_preview["image_obs"]
                    if isinstance(img_seq, dict):
                        img_seq = img_seq[sorted(img_seq.keys())[0]]
                    seq = img_seq[0]
                    # Compute metrics on the preview
                    state_seq, image_seq, actions = prepare_batch(
                        val_preview, self.device, self.args.modality_type
                    )
                    with torch.amp.autocast(
                        device_type="cuda",
                        enabled=self.amp,
                        dtype=getattr(self, "amp_dtype", torch.float16),
                    ):
                        pred = self.model.sample_actions(
                            state_seq, image_seq, steps=self.args.val_flow_steps
                        )
                    metrics = self._compute_metrics(pred, actions)
                    # Merge aggregate mse/l2 with preview-based baseline metrics
                    metrics["mse"] = val_metrics["mse"]
                    metrics["l2"] = val_metrics["l2"]
                    metrics["cosine"] = val_metrics["cosine"]
                    self._log_grid("val", seq, metrics, epoch)

                # Best checkpoint
                if val_metrics is not None and val_metrics["mse"] < self.best_val:
                    self.best_val = val_metrics["mse"]
                    best_path = os.path.join(self.args.ckpt_dir, "best.pt")
                    self.save_checkpoint(best_path, epoch, self.best_val)
                    if self.run is not None:
                        try:
                            art = wandb.Artifact(
                                name=f"{self.args.run_name}-model",
                                type="model",
                                metadata={"best_val": self.best_val},
                            )
                            art.add_file(best_path)
                            if self.ds.action_stats is not None:
                                art.metadata.update(
                                    {
                                        "action_min": float(
                                            self.ds.action_stats["min"]
                                        ),
                                        "action_max": float(
                                            self.ds.action_stats["max"]
                                        ),
                                    }
                                )
                            wandb.log_artifact(art)
                        except Exception:
                            pass

            # Periodic checkpoint
            if epoch % max(10, self.args.eval_freq) == 0:
                self.save_checkpoint(
                    os.path.join(self.args.ckpt_dir, f"epoch_{epoch}.pt"),
                    epoch,
                    self.best_val if val_metrics is None else val_metrics["mse"],
                )

            # Epoch summary
            epoch_time = time.time() - epoch_start
            if self.run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train/epoch_loss": float(train_loss),
                    "best_val/loss": float(self.best_val),
                    "time/epoch_sec": float(epoch_time),
                }
                if val_metrics is not None:
                    log_dict.update(
                        {
                            "val/loss": float(val_metrics["mse"]),
                            "val/l2": float(val_metrics["l2"]),
                            "val/cosine": float(val_metrics["cosine"]),
                        }
                    )
                wandb.log(log_dict, step=self.global_step)

            # Train preview grid + metrics
            if train_preview is not None and self.args.modality_type in (
                "image",
                "state+image",
            ):
                img_seq = train_preview["image_obs"]
                if isinstance(img_seq, dict):
                    img_seq = img_seq[sorted(img_seq.keys())[0]]
                seq = img_seq[0]
                with torch.no_grad():
                    state_seq, image_seq, actions = prepare_batch(
                        train_preview, self.device, self.args.modality_type
                    )
                    with torch.amp.autocast(
                        device_type="cuda",
                        enabled=self.amp,
                        dtype=getattr(self, "amp_dtype", torch.float16),
                    ):
                        pred = self.model.sample_actions(
                            state_seq, image_seq, steps=self.args.val_flow_steps
                        )
                    metrics = self._compute_metrics(pred, actions)
                self._log_grid("train", seq, metrics, epoch)

            print(
                f"Epoch {epoch:04d} | train_loss={train_loss:.4f} | best_val={self.best_val:.4f} | time={epoch_time:.1f}s"
            )

            # Step LR scheduler at epoch end
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

        if self.run is not None:
            wandb.finish()


def main():
    args = load_args(Args)
    Trainer(args).fit()


if __name__ == "__main__":
    main()
