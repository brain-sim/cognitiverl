"""Clean Training Pipeline for Imitation Learning with Flow Matching.

Inspired by cleanRL/leanRL for simplicity and PyTorch Lightning for logging.
Supports train, validation, and play phases with comprehensive metrics.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# IsaacLab imports
from isaaclab.app import AppLauncher
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

# Local imports
from scripts.imitationrl.dataset import SequenceDataset
from scripts.imitationrl.models import SeqFlowPolicy, VanillaFlowPolicy
from scripts.utils import load_args, make_isaaclab_env, seed_everything

try:
    import wandb
except ImportError:
    wandb = None

# WandB configuration
WANDB_DIR_DEFAULT = "/home/chandramouli/cognitiverl"


@dataclass
class TrainArgs:
    """Training configuration - cleanRL style."""

    # Core training
    device: str = "cuda:0"
    seed: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    eval_freq: int = 1
    save_freq: int = 1

    # Learning rate scheduling - ADD THESE
    lr_decay: float = 0.5  # Changed default from 1.0 to 0.5 for actual decay
    lr_steps: List[int] = None  # Will default to [10, 20, 50]

    # Data
    dataset: str = (
        "/home/chandramouli/cognitiverl/datasets/generated_dataset_gr1_nut_pouring.hdf5"
    )
    train_split: float = 0.8
    sequence_length: int = 10
    pad_sequence: bool = True
    normalize_actions: bool = True
    demo_limit: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True

    # Model Architecture - ADD THESE MISSING PARAMETERS
    modality_type: str = "state+image"  # "state", "image", "state+image"
    state_keys: List[str] = None
    image_keys: List[str] = None
    img_size: Tuple[int, int, int] = (3, 160, 256)  # (C, H, W)

    # Model hyperparameters (ADD THESE)
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    ff_hidden: int = 512
    dropout: float = 0.1

    # Flow matching
    flow_steps: int = 10
    val_flow_steps: int = 10

    # Logging
    log: bool = False  # True=online, False=offline
    wandb_project: str = "bc"
    wandb_dir: str = WANDB_DIR_DEFAULT
    run_name: str = "seq_flow_bc"
    log_image_freq: int = 1  # Log images every N epochs

    resume: Optional[str] = None

    # Environment (for play)
    task: str = "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0"
    num_eval_envs: int = 16
    num_eval_steps: int = 350
    headless: bool = True
    enable_cameras: bool = True
    capture_video: bool = True

    # Mixed precision
    amp: bool = True
    amp_dtype: str = "bf16"


def get_args():
    return load_args(TrainArgs)


# ============================================================================
# Utility Functions (reusable across modalities)
# ============================================================================


class ObservationHandler:
    """Clean handler for different observation modalities."""

    def __init__(
        self, modality_type: str, state_keys: List[str], image_keys: List[str]
    ):
        self.modality_type = modality_type
        self.state_keys = state_keys or []
        self.image_keys = image_keys or []

    def process_batch(
        self, batch: Dict, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Process batch into (state_seq, image_seq, actions) - reusable function."""
        # Use dataset's process_batch method
        return batch  # Dataset already processes this

    def log_shapes(self, state_seq, image_seq, actions, prefix=""):
        """Log tensor shapes for debugging."""
        shapes = {}
        if state_seq is not None:
            shapes[f"{prefix}state_shape"] = list(state_seq.shape)
        if image_seq is not None:
            shapes[f"{prefix}image_shape"] = list(image_seq.shape)
        shapes[f"{prefix}action_shape"] = list(actions.shape)
        return shapes


def compute_metrics(
    pred_actions: torch.Tensor, target_actions: torch.Tensor
) -> Dict[str, float]:
    """Compute MSE and cosine similarity between predictions and targets."""
    mse = F.mse_loss(pred_actions, target_actions).item()

    # Cosine similarity (higher is better)
    pred_flat = pred_actions.view(pred_actions.size(0), -1)
    target_flat = target_actions.view(target_actions.size(0), -1)
    cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean().item()

    return {"mse": mse, "cosine_similarity": cosine_sim}


def compute_random_baseline(target_actions: torch.Tensor) -> Dict[str, float]:
    """Compute metrics against random baseline."""
    random_actions = torch.randn_like(target_actions)
    return compute_metrics(random_actions, target_actions)


def make_image_grid(
    images: torch.Tensor, nrow: int = 4, max_images: int = 8
) -> torch.Tensor:
    """Create image grid for logging (handles different formats).

    For sequence data: creates grid with samples as rows and timesteps as columns.
    """
    if images.dim() == 5:  # (B, T, C, H, W) - sequence format
        B, T, C, H, W = images.shape
        num_samples = min(max_images, B)

        # Select samples and reshape for grid: (num_samples * T, C, H, W)
        selected_images = images[:num_samples]  # (num_samples, T, C, H, W)

        # Reshape to show samples as rows, timesteps as columns
        grid_images = selected_images.view(num_samples * T, C, H, W)

        # Create grid with T columns (timesteps) and num_samples rows
        grid = torchvision.utils.make_grid(
            grid_images,
            nrow=T,  # T timesteps per row
            normalize=True,
            value_range=(0, 1) if grid_images.max() <= 1.0 else None,
        )
        return grid

    elif images.dim() == 4:  # (B, C, H, W) - single timestep
        if images.size(1) > 8:  # Likely (B, T*C, H, W) - need to reshape
            B, TxC, H, W = images.shape
            T = TxC // 3  # Assume 3 channels
            images = images.view(B, T, 3, H, W)
            # Now treat as sequence format
            return make_image_grid(images, nrow=nrow, max_images=max_images)
        else:
            # Regular (B, C, H, W) format
            images = images[:max_images]

    # Normalize to [0, 1] if needed
    if images.max() > 1.0:
        images = images / 255.0

    return torchvision.utils.make_grid(images, nrow=nrow, normalize=True)


def print_section_header(title: str, char: str = "=", width: int = 80):
    """Print a nice section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def format_number(num):
    """Format large numbers with K/M suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


# ============================================================================
# Main Trainer Class
# ============================================================================


class FlowBCTrainer:
    """Clean trainer class inspired by cleanRL with PyTorch Lightning logging."""

    def __init__(self, args: TrainArgs):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        seed_everything(args.seed, use_torch=True, torch_deterministic=True)

        # Setup training state first
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.epoch_metrics = {"train": [], "val": []}

        # Mixed precision setup (before model info)
        self.amp_enabled = (self.device.type == "cuda") and bool(args.amp)
        self.amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        # Initialize logging
        self.run = self._init_wandb()

        # Setup data
        self.dataset = self._setup_dataset()
        self.train_loader, self.val_loader = self._setup_dataloaders()

        # Print comprehensive dataset info
        self._print_dataset_info()

        # Setup model (now returns scheduler too)
        self.model, self.optimizer, self.scheduler = self._setup_model()

        # Print model info (now that everything is set up)
        self._print_model_info()

        # Observation handler
        self.obs_handler = ObservationHandler(
            args.modality_type, args.state_keys, args.image_keys
        )

        # Setup environment for play
        self._setup_environment()

        # Resume from checkpoint if specified
        if args.resume:
            self._load_checkpoint(args.resume)

    def _print_dataset_info(self):
        """Print comprehensive dataset information."""
        print_section_header("📊 DATASET INFORMATION")

        # Basic info
        print(f"📂 Dataset path: {self.args.dataset}")
        print(f"📦 Total sequences: {format_number(len(self.dataset))}")
        print(f"🔄 Sequence length: {self.args.sequence_length}")
        print(f"📋 Modality type: {self.args.modality_type}")
        print(f"🎯 Normalize actions: {self.args.normalize_actions}")

        # Train/val split info
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)
        print(
            f"📈 Train samples: {format_number(train_size)} ({self.args.train_split:.1%})"
        )
        print(
            f"📊 Val samples: {format_number(val_size)} ({1 - self.args.train_split:.1%})"
        )

        # DataLoader info
        print(f"🔧 Batch size: {self.args.batch_size}")
        print(f"👥 Num workers: {self.args.num_workers}")
        print(f"📌 Pin memory: {self.args.pin_memory}")
        print("🎲 Shuffle train: True")

        # Sample data info
        sample = self.dataset[0]
        state_seq, image_seq, actions = self.dataset.process_batch(
            {
                k: torch.from_numpy(np.array([v]))
                if isinstance(v, np.ndarray)
                else {
                    obs_k: torch.from_numpy(np.array([obs_v]))
                    for obs_k, obs_v in v.items()
                }
                for k, v in sample.items()
            },
            self.device,
        )

        print("\n📏 Data Shapes & Types:")
        if state_seq is not None:
            print(
                f"  🔢 State: {list(state_seq.shape)} | {state_seq.dtype} | {state_seq.device}"
            )
        if image_seq is not None:
            print(
                f"  🖼️  Image: {list(image_seq.shape)} | {image_seq.dtype} | {image_seq.device}"
            )
        print(
            f"  🎯 Action: {list(actions.shape)} | {actions.dtype} | {actions.device}"
        )

        # Action normalization info
        if self.action_normalizer:
            stats = self.action_normalizer.get_stats()
            print("\n🎯 Action Normalization:")
            print(f"  📊 Original range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print("  🔄 Normalized to: [-1.0, 1.0]")

        # Keys info
        if self.args.modality_type in ("state", "state+image"):
            print(f"\n🔑 State keys: {self.dataset.state_keys}")
        if self.args.modality_type in ("image", "state+image"):
            print(f"🖼️  Image keys: {self.dataset.image_keys}")

    def _print_model_info(self):
        """Print comprehensive model information."""
        print_section_header("🤖 MODEL INFORMATION")

        # Model architecture
        print("🏗️  Model: SeqFlowPolicy")
        print(f"🎯 Device: {self.device}")

        # Mixed precision info (with safe access)
        amp_enabled = getattr(self, "amp_enabled", False)
        amp_dtype = getattr(self, "amp_dtype", "N/A")
        print(
            f"💨 Mixed precision: {amp_enabled} ({amp_dtype if amp_enabled else 'N/A'})"
        )

        # Model parameters - ADD DEBUG PRINTS
        try:
            print("🔧 Computing model parameters...")
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f"📊 Total parameters: {format_number(total_params)}")
            print(f"🎯 Trainable parameters: {format_number(trainable_params)}")
        except Exception as e:
            print(f"❌ Error computing parameters: {e}")

        # Optimizer and scheduler info
        try:
            print("🔧 Getting optimizer info...")
            print(f"⚡ Optimizer: {self.optimizer.__class__.__name__}")
            print(f"📈 Learning rate: {self.args.learning_rate}")

            # Scheduler info
            lr_steps = self.args.lr_steps or [10, 20, 50]
            print("📉 LR Scheduler: MultiStepLR")
            print(f"📅 LR decay steps: {lr_steps}")
            print(f"📉 LR decay factor: {self.args.lr_decay}")
            print(f"🔄 Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        except Exception as e:
            print(f"❌ Error getting optimizer info: {e}")

        # Flow matching info
        print(f"🌊 Flow steps (train): {self.args.flow_steps}")
        print(f"🔬 Flow steps (val): {self.args.val_flow_steps}")

        # Model structure (simplified) - ADD DEBUG
        try:
            print("🔧 Computing model structure...")
            print("\n🏛️  Model Structure:")
            for name, module in self.model.named_children():
                if hasattr(module, "parameters"):
                    module_params = sum(p.numel() for p in module.parameters())
                    print(f"  📦 {name}: {format_number(module_params)} params")
        except Exception as e:
            print(f"❌ Error computing model structure: {e}")

    def _init_wandb(self):
        """Initialize WandB logging."""
        if wandb is None:
            return None

        try:
            os.makedirs(self.args.wandb_dir, exist_ok=True)
        except Exception:
            pass

        os.environ["WANDB_MODE"] = "online" if self.args.log else "offline"
        os.environ["WANDB_DIR"] = self.args.wandb_dir

        try:
            run = wandb.init(
                project=self.args.wandb_project,
                name=self.args.run_name,
                config=vars(self.args),
                dir=self.args.wandb_dir,
                save_code=True,
            )

            # Create wandb subdirectories
            run_dir = run.dir
            self.ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.videos_dir = os.path.join(run_dir, "videos")
            os.makedirs(self.videos_dir, exist_ok=True)

            return run
        except Exception as e:
            print(f"Warning: WandB initialization failed: {e}")
            return None

    def _setup_dataset(self):
        """Setup dataset with action normalization."""
        # Default keys
        state_keys = self.args.state_keys or [
            "left_eef_pos",
            "hand_joint_state",
            "right_eef_pos",
            "right_eef_quat",
            "left_eef_quat",
        ]
        image_keys = self.args.image_keys or ["robot_pov_cam"]

        dataset = SequenceDataset(
            hdf5_path=self.args.dataset,
            state_keys=state_keys,
            image_keys=image_keys,
            sequence_length=self.args.sequence_length,
            pad_sequence=self.args.pad_sequence,
            normalize_actions=self.args.normalize_actions,
            modality_type=self.args.modality_type,
        )

        self.action_normalizer = dataset.get_action_normalizer()
        return dataset

    def _setup_dataloaders(self):
        """Setup train/val dataloaders with 80-20 split."""
        train_size = int(self.args.train_split * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.args.seed),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=False,
        )

        return train_loader, val_loader

    def _setup_model(self):
        """Setup model and optimizer."""
        # Infer dimensions from sample - Fix the processing
        try:
            sample = self.dataset[0]

            # Create a proper batch format for processing
            if isinstance(sample.get("obs"), dict):
                # Handle robomimic format
                obs_batch = {}
                for key, value in sample["obs"].items():
                    obs_batch[key] = torch.from_numpy(np.array([value])).to(self.device)
                batch_sample = {
                    "obs": obs_batch,
                    "actions": torch.from_numpy(np.array([sample["actions"]])).to(
                        self.device
                    ),
                }
            else:
                # Handle other formats
                batch_sample = {
                    k: torch.from_numpy(np.array([v]))
                    if isinstance(v, np.ndarray)
                    else v
                    for k, v in sample.items()
                }

            state_seq, image_seq, actions = self.dataset.process_batch(
                batch_sample, self.device
            )

            action_dim = actions.shape[-1]
            state_dim = state_seq.shape[-1] if state_seq is not None else 0

            # Update img_size from actual data
            if image_seq is not None:
                self.args.img_size = image_seq.shape[-3:]

            print(
                f"✅ Inferred dimensions: state_dim={state_dim}, action_dim={action_dim}"
            )

        except Exception as e:
            print(f"❌ Error inferring dimensions: {e}")
            # Fallback to defaults
            action_dim = 36  # From your terminal output
            state_dim = 36  # From your terminal output
            print(
                f"🔧 Using fallback dimensions: state_dim={state_dim}, action_dim={action_dim}"
            )

        print("🔧 Creating model...")
        try:
            model = VanillaFlowPolicy(state_dim, action_dim, self.args).to(self.device)
            print("✅ Model created successfully")
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            raise

        print("🔧 Creating optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate)
        print("✅ Optimizer created successfully")

        # Setup learning rate scheduler
        lr_steps = self.args.lr_steps or [10, 20, 50]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_steps, gamma=self.args.lr_decay
        )

        print("✅ Optimizer and scheduler created successfully")
        print(f"📈 LR schedule: decay by {self.args.lr_decay}x at epochs {lr_steps}")

        return model, optimizer, scheduler

    def _setup_environment(self):
        """Setup IsaacLab environment for play."""
        print_section_header("🎮 ENVIRONMENT SETUP")
        try:
            print("🔧 Importing required modules...")
            import isaaclab_tasks  # noqa: F401
            import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

            print("✅ Modules imported successfully")

            print("🔧 Creating environment...")
            self.eval_envs = make_isaaclab_env(
                self.args.seed,
                self.args.task,
                self.args.device,
                self.args.num_eval_envs,
                self.args.capture_video,
                False,  # disable_fabric
                log_dir=self.run.dir if self.run else ".",
                video_length=self.args.num_eval_steps,
                video_interval=self.args.num_eval_steps,
            )()

            print(f"🎯 Task: {self.args.task}")
            print(f"🤖 Num environments: {self.args.num_eval_envs}")
            print(f"📹 Capture video: {self.args.capture_video}")
            print(f"👁️  Headless: {self.args.headless}")
            print(f"📸 Enable cameras: {self.args.enable_cameras}")
            print("✅ Environment initialized successfully")

        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            print("🔧 Continuing without environment (play will be skipped)")
            self.eval_envs = None

    def _save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Save checkpoint with normalization values."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),  # Save scheduler state
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "args": vars(self.args),
            "action_normalizer": self.action_normalizer.get_stats()
            if self.action_normalizer
            else None,
        }

        torch.save(checkpoint, path)

        if self.run:
            # Log checkpoint as artifact
            artifact = wandb.Artifact(
                name=f"checkpoint_epoch_{epoch}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}",
            )
            artifact.add_file(path)
            self.run.log_artifact(artifact)

    def _load_checkpoint(self, path: str):
        """Load checkpoint and restore training state."""
        print(f"🔄 Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        # Restore action normalizer if available
        if "action_normalizer" in checkpoint and checkpoint["action_normalizer"]:
            from scripts.imitationrl.dataset import ActionNormalizer

            self.action_normalizer = ActionNormalizer.from_stats(
                checkpoint["action_normalizer"]
            )

        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"✅ Resumed from epoch {self.start_epoch}, best_val_loss={self.best_val_loss:.4f}"
        )
        print(f"📈 Current learning rate: {current_lr:.2e}")

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step with flow matching loss."""
        self.model.train()

        # Process batch
        state_seq, image_seq, actions = self.dataset.process_batch(batch, self.device)

        # Forward pass with mixed precision
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            # Flow matching loss
            flow_loss = self.model.compute_flow_loss(state_seq, image_seq, actions)

            # Sample actions for metrics (no gradients)
            with torch.no_grad():
                pred_actions = self.model.sample_actions(
                    state_seq, image_seq, steps=self.args.flow_steps
                )

        # Backward pass
        self.scaler.scale(flow_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(pred_actions, actions)
            random_metrics = compute_random_baseline(actions)

            step_metrics = {
                "train_step/flow_loss": flow_loss.item(),
                "train_step/pred_mse": metrics["mse"],
                "train_step/pred_cosine_sim": metrics["cosine_similarity"],
                "train_step/random_mse": random_metrics["mse"],
                "train_step/random_cosine_sim": random_metrics["cosine_similarity"],
            }

        return step_metrics

    @torch.no_grad()
    def val_step(self, batch: Dict) -> Dict[str, float]:
        """Single validation step with action sampling."""
        self.model.eval()

        # Process batch
        state_seq, image_seq, actions = self.dataset.process_batch(batch, self.device)

        with torch.amp.autocast(
            device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            # Sample actions (main validation metric)
            pred_actions = self.model.sample_actions(
                state_seq, image_seq, steps=self.args.val_flow_steps
            )

        # Compute metrics
        metrics = compute_metrics(pred_actions, actions)
        random_metrics = compute_random_baseline(actions)

        step_metrics = {
            "val_step/pred_mse": metrics["mse"],
            "val_step/pred_cosine_sim": metrics["cosine_similarity"],
            "val_step/random_mse": random_metrics["mse"],
            "val_step/random_cosine_sim": random_metrics["cosine_similarity"],
        }

        return step_metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training epoch with Lightning-style progress display."""
        self.model.train()
        epoch_metrics = []

        # Log image grid at start of epoch
        if epoch % self.args.log_image_freq == 0 and self.args.modality_type in (
            "image",
            "state+image",
        ):
            self._log_image_grid(epoch, "train_epoch")

        # Lightning-style progress bar with better info
        pbar = tqdm(
            self.train_loader,
            desc="  🔥 Training",
            leave=False,
            ncols=140,
            position=1,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        running_loss = 0.0
        running_mse = 0.0
        running_cos_sim = 0.0

        for step, batch in enumerate(pbar):
            # Training step
            step_metrics = self.train_step(batch)
            epoch_metrics.append(step_metrics)

            # Update running averages with more recent bias
            alpha = 0.1  # smoothing factor
            running_loss = (1 - alpha) * running_loss + alpha * step_metrics[
                "train_step/flow_loss"
            ]
            running_mse = (1 - alpha) * running_mse + alpha * step_metrics[
                "train_step/pred_mse"
            ]
            running_cos_sim = (1 - alpha) * running_cos_sim + alpha * step_metrics[
                "train_step/pred_cosine_sim"
            ]

            # Step-wise logging
            if self.run:
                self.run.log(
                    step_metrics, step=(epoch + 1) * len(self.train_loader) + step
                )

            # Update progress bar with key metrics
            pbar.set_postfix(
                {
                    "loss": f"{running_loss:.4f}",
                    "mse": f"{running_mse:.4f}",
                    "cos_sim": f"{running_cos_sim:.3f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.1e}",
                    "grad_norm": f"{self._get_grad_norm():.2f}",
                }
            )

        # Epoch-wise averaging (PyTorch Lightning style)
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key.replace("step", "epoch")] = np.mean(
                [m[key] for m in epoch_metrics]
            )

        return avg_metrics

    def _get_grad_norm(self) -> float:
        """Get gradient norm for monitoring."""
        try:
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** (1.0 / 2)
        except:
            return 0.0

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> Dict[str, float]:
        """Validation epoch with Lightning-style progress display."""
        self.model.eval()
        epoch_metrics = []

        # Log image grid
        if epoch % self.args.log_image_freq == 0 and self.args.modality_type in (
            "image",
            "state+image",
        ):
            self._log_image_grid(epoch, "val")

        # Lightning-style progress bar
        pbar = tqdm(
            self.val_loader,
            desc="  🔍 Validating",
            leave=False,
            ncols=140,
            position=1,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        running_mse = 0.0
        running_cos_sim = 0.0

        for step, batch in enumerate(pbar):
            # Validation step
            step_metrics = self.val_step(batch)
            epoch_metrics.append(step_metrics)

            # Update running averages
            alpha = 0.1
            running_mse = 0.9 * running_mse + 0.1 * step_metrics["val_step/pred_mse"]
            running_cos_sim = (
                0.9 * running_cos_sim + 0.1 * step_metrics["val_step/pred_cosine_sim"]
            )

            # Update progress bar
            pbar.set_postfix(
                {
                    "val_mse": f"{running_mse:.4f}",
                    "val_cos_sim": f"{running_cos_sim:.3f}",
                }
            )

        # Epoch-wise averaging
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key.replace("step", "epoch")] = np.mean(
                [m[key] for m in epoch_metrics]
            )

        return avg_metrics

    def _log_image_grid(self, epoch: int, split: str):
        """Log image grid for visualization with samples as rows, timesteps as columns."""
        if self.args.modality_type not in ("image", "state+image"):
            return

        # Get a batch
        loader = self.train_loader if split == "train" else self.val_loader
        batch = next(iter(loader))

        # Process batch
        state_seq, image_seq, actions = self.dataset.process_batch(batch, self.device)

        if image_seq is None:
            return

        # Select random subset (max 8 samples)
        batch_size = image_seq.size(0)
        num_samples = min(8, batch_size)
        indices = torch.randperm(batch_size)[:num_samples]

        selected_images = image_seq[indices]  # (num_samples, T, C, H, W)

        print(
            f"🖼️  Creating image grid: {num_samples} samples × {selected_images.size(1)} timesteps"
        )

        # Create grid: samples as rows, timesteps as columns
        image_grid = make_image_grid(selected_images, max_images=num_samples)

        if self.run:
            self.run.log(
                {
                    f"{split}/image_grid": wandb.Image(
                        image_grid,
                        caption=f"Epoch {epoch} - {split} samples (rows) × timesteps (cols)",
                    )
                },
                step=epoch,
            )

        print(
            f"✅ Logged {split} image grid: {num_samples} samples × {selected_images.size(1)} timesteps"
        )

    @torch.no_grad()
    def play_in_environment(self, epoch: int):
        """Play model in IsaacLab environment with video recording."""
        if self.eval_envs is None:
            print("⚠️  Environment not available, skipping play")
            return

        print(f"🎮 Playing model in environment (epoch {epoch})")

        # Set model to eval mode
        self.model.eval()

        # Reset environment
        obs, _ = self.eval_envs.reset()
        step_count = 0
        done = torch.zeros(self.args.num_eval_envs, dtype=torch.bool)

        episode_rewards = []
        current_rewards = torch.zeros(self.args.num_eval_envs)

        # Initialize rolling buffers for sequence (similar to robomimic padding)
        state_buffer, image_buffer = self._init_sequence_buffers(obs)

        play_pbar = tqdm(
            range(self.args.num_eval_steps), desc="🎮 Playing", leave=False, ncols=100
        )

        for step in play_pbar:
            if done.all():
                break

            # Update rolling buffers with current observation
            state_buffer, image_buffer = self._update_sequence_buffers(
                obs, state_buffer, image_buffer
            )

            # Get action from model using current sequence buffers
            with torch.amp.autocast(
                device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
            ):
                pred_actions = self.model.sample_actions(
                    state_buffer,
                    image_buffer,
                    steps=self.args.val_flow_steps,
                    deterministic=True,
                )

            # Denormalize actions for environment
            if self.action_normalizer:
                pred_actions = self.action_normalizer.denormalize(pred_actions)

            # Step environment
            obs, rewards, done, _ = self.eval_envs.step(pred_actions)

            # Track rewards
            current_rewards += rewards.cpu()

            # Handle episode completion - reset buffers for completed episodes
            for i in range(self.args.num_eval_envs):
                if done[i]:
                    if len(episode_rewards) < 100:  # Limit logging
                        episode_rewards.append(current_rewards[i].item())
                    current_rewards[i] = 0.0
                    # Reset sequence buffers for this environment
                    if state_buffer is not None:
                        state_buffer[i] = 0.0
                    if image_buffer is not None:
                        image_buffer[i] = 0.0

            step_count += 1
            play_pbar.set_postfix(
                {
                    "completed": f"{done.sum().item()}/{self.args.num_eval_envs}",
                    "avg_reward": f"{np.mean(episode_rewards):.2f}"
                    if episode_rewards
                    else "N/A",
                }
            )

        # Log play results
        if episode_rewards and self.run:
            play_metrics = {
                "play/avg_reward": np.mean(episode_rewards),
                "play/max_reward": np.max(episode_rewards),
                "play/min_reward": np.min(episode_rewards),
                "play/episodes_completed": len(episode_rewards),
                "play/steps_taken": step_count,
            }
            self.run.log(play_metrics, step=(epoch + 1) * len(self.train_loader))

        print(
            f"🎯 Play completed: {len(episode_rewards)} episodes, avg reward: {np.mean(episode_rewards):.2f}"
        )

    def _init_sequence_buffers(self, initial_obs):
        """Initialize rolling sequence buffers with zeros (like robomimic padding)."""
        batch_size = self.args.num_eval_envs
        seq_len = self.args.sequence_length

        state_buffer = None
        image_buffer = None

        try:
            # Initialize state buffer if needed
            if self.args.modality_type in ("state", "state+image"):
                # Infer state dimension from first observation
                state_parts = []
                for key in self.dataset.state_keys:
                    if key in initial_obs:
                        obs_data = initial_obs[key]
                        flat_dim = obs_data.view(batch_size, -1).shape[1]
                        state_parts.append(flat_dim)

                if state_parts:
                    total_state_dim = sum(state_parts)
                    state_buffer = torch.zeros(
                        batch_size,
                        seq_len,
                        total_state_dim,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    print(f"🔧 Initialized state buffer: {state_buffer.shape}")

            # Initialize image buffer if needed
            if self.args.modality_type in ("image", "state+image"):
                # Use first available image key
                for key in self.dataset.image_keys:
                    if key in initial_obs:
                        img_data = initial_obs[key]
                        if img_data.dim() == 4:  # (B, C, H, W)
                            _, H, W, C = img_data.shape
                            image_buffer = torch.zeros(
                                batch_size,
                                seq_len,
                                C,
                                H,
                                W,
                                device=self.device,
                                dtype=torch.float32,
                            )
                            print(f"🔧 Initialized image buffer: {image_buffer.shape}")
                            break

        except Exception as e:
            print(f"⚠️  Error initializing buffers: {e}")
            print("🔧 Using fallback buffer initialization")

            # Fallback initialization
            if self.args.modality_type in ("state", "state+image"):
                state_buffer = torch.zeros(
                    batch_size,
                    seq_len,
                    36,  # fallback state dim
                    device=self.device,
                    dtype=torch.float32,
                )

            if self.args.modality_type in ("image", "state+image"):
                C, H, W = self.args.img_size
                image_buffer = torch.zeros(
                    batch_size,
                    seq_len,
                    C,
                    H,
                    W,
                    device=self.device,
                    dtype=torch.float32,
                )
        return state_buffer, image_buffer

    def _update_sequence_buffers(self, obs, state_buffer, image_buffer):
        """Update rolling sequence buffers with new observation (robomimic-style rolling)."""

        try:
            # Update state buffer
            if state_buffer is not None and self.args.modality_type in (
                "state",
                "state+image",
            ):
                # Extract current state observation
                state_parts = []
                for key in self.dataset.state_keys:
                    if key in obs:
                        obs_data = obs[key]
                        flat_data = obs_data.view(
                            obs_data.shape[0], -1
                        )  # (B, features)
                        state_parts.append(flat_data)

                if state_parts:
                    current_state = torch.cat(
                        state_parts, dim=-1
                    )  # (B, total_state_dim)

                    # Roll buffer: move everything left by 1, add new obs at the end
                    state_buffer = torch.roll(state_buffer, shifts=-1, dims=1)
                    state_buffer[:, -1, :] = current_state.to(self.device)

            # Update image buffer
            if image_buffer is not None and self.args.modality_type in (
                "image",
                "state+image",
            ):
                # Extract current image observation (use first available key)
                for key in self.dataset.image_keys:
                    if key in obs:
                        img_data = obs[key]
                        if img_data.dim() == 4:  # (B, C, H, W)
                            # Normalize to [0, 1] if needed
                            if img_data.max() > 1.0:
                                img_data = img_data.float() / 255.0

                            # Roll buffer: move everything left by 1, add new obs at the end
                            image_buffer = torch.roll(image_buffer, shifts=-1, dims=1)
                            image_buffer[:, -1, :, :, :] = img_data.permute(
                                0, 3, 1, 2
                            ).to(self.device)
                            break

        except Exception as e:
            print(f"⚠️  Error updating buffers: {e}")
            # Continue with existing buffers (graceful degradation)

        return state_buffer, image_buffer

    def fit(self):
        """Main training loop with comprehensive logging."""
        print_section_header("🚀 TRAINING STARTED")
        print(f"📅 Epochs: {self.start_epoch + 1} → {self.args.num_epochs}")
        print(f"💾 Checkpoint dir: {self.ckpt_dir}")
        print(f"📊 WandB: {'Online' if self.args.log else 'Offline'}")
        print(f"🎮 Environment play: Every {self.args.eval_freq * 2} epochs")

        for epoch in range(self.start_epoch + 1, self.args.num_epochs + 1):
            epoch_start = time.time()

            # Get current learning rate before training
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % self.args.eval_freq == 0:
                val_metrics = self.val_epoch(epoch)

                # Check for best model
                val_loss = val_metrics["val_epoch/pred_mse"]
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                # Play in environment after validation
                if epoch % self.args.eval_freq == 0:  # Less frequent play
                    self.play_in_environment(epoch)
            else:
                val_metrics = {}
                is_best = False

            # Step the learning rate scheduler
            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]

            # Log learning rate change if it happened
            if old_lr != new_lr:
                print(f"📉 Learning rate decayed: {old_lr:.2e} → {new_lr:.2e}")

            # Epoch logging
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["epoch"] = epoch
            epoch_metrics["best_val_loss"] = self.best_val_loss
            epoch_metrics["epoch_time"] = time.time() - epoch_start
            epoch_metrics["learning_rate"] = new_lr  # Log current learning rate

            if self.run:
                self.run.log(epoch_metrics, step=(epoch + 1) * len(self.train_loader))

            # Save checkpoints
            if epoch % self.args.save_freq == 0:
                ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
                self._save_checkpoint(ckpt_path, epoch, is_best=False)

            if is_best:
                best_path = os.path.join(self.ckpt_dir, "best.pt")
                self._save_checkpoint(best_path, epoch, is_best=True)

            # Print epoch summary (Lightning style)
            metrics_str = []
            for k, v in epoch_metrics.items():
                if isinstance(v, float) and "time" not in k and k != "learning_rate":
                    metrics_str.append(f"{k}: {v:.4f}")

            status_emoji = "🏆" if is_best else "✅"
            lr_info = f"lr: {new_lr:.1e}" if new_lr != current_lr else ""
            print(
                f"{status_emoji} Epoch {epoch:3d} | "
                + " | ".join(metrics_str)
                + (f" | 📈 {lr_info}" if lr_info else "")
                + f" | ⏱️  {epoch_metrics['epoch_time']:.1f}s"
            )

        if self.run:
            self.run.finish()

        print_section_header("🎉 TRAINING COMPLETED")

    def __del__(self):
        """Cleanup environment on deletion."""
        if hasattr(self, "eval_envs") and self.eval_envs is not None:
            try:
                self.eval_envs.close()
            except:
                pass


def main():
    """Main training function."""
    args = get_args()

    # Launch IsaacLab app
    try:
        from argparse import Namespace

        import pinocchio  # noqa: F401

        app_launcher = AppLauncher(
            Namespace(
                **{
                    "headless": args.headless,
                    "enable_cameras": args.enable_cameras,
                }
            )
        )
        simulation_app = app_launcher.app
        print("✅ IsaacLab app launched successfully")
    except ImportError:
        print("❌ IsaacLab not available")
        simulation_app = None

    try:
        # Create and run trainer
        trainer = FlowBCTrainer(args)
        trainer.fit()
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print("Exception:", e)
    finally:
        if simulation_app:
            simulation_app.close()
            print("🔄 IsaacLab app closed")


if __name__ == "__main__":
    main()
