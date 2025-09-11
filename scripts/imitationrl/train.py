"""Clean Training Pipeline for Imitation Learning with Flow Matching.

Inspired by cleanRL/leanRL for simplicity and PyTorch Lightning for logging.
Supports train, validation, and play phases with comprehensive metrics.
Refactored to use TensorDict extensively for cleaner code.
"""

import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# IsaacLab imports
from isaaclab.app import AppLauncher
from tensordict import TensorDict
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

# Local imports
from scripts.imitationrl.dataset import SequenceDataset
from scripts.imitationrl.models import BCPolicy
from scripts.imitationrl.utils import (
    compute_metrics,
    compute_random_baseline,
    format_number,
    get_grad_norm,
    make_image_grid,
    print_section_header,
    process_image_batch,
)
from scripts.utils import load_args, make_isaaclab_env, seed_everything

try:
    import wandb
except ImportError:
    wandb = None

# WandB configuration
WANDB_DIR_DEFAULT = "/home/chandramouli/cognitiverl"

os.environ["WANDB_IGNORE_GLOBS"] = "*.pt,*.mp4"


@dataclass
class TrainArgs:
    """Training configuration - cleanRL style."""

    # Core training
    device: str = "cuda:0"
    seed: int = 1
    batch_size: int = 32
    learning_rate: float = 5e-4
    num_epochs: int = 100
    eval_freq: int = 1
    save_freq: int = 1

    # Add validation flag
    validation: bool = False  # New flag to enable/disable validation

    # Learning rate scheduling - ADD THESE
    lr_decay: float = 0.5  # Changed default from 1.0 to 0.5 for actual decay
    lr_steps: Tuple[int] = (10, 20, 50)  # Will default to [10, 20, 50]

    # Data
    dataset: str = (
        "/home/chandramouli/cognitiverl/datasets/generated_dataset_gr1_nut_pouring.hdf5"
    )
    train_split: float = 1.0
    sequence_length: int = 1
    pad_sequence: bool = True
    frame_stack: int = 10
    pad_frame_stack: bool = True
    normalize_actions: bool = True
    demo_limit: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True

    # Model Architecture - ADD THESE MISSING PARAMETERS
    modality_type: str = "state+image"  # "state", "image", "state+image"
    state_keys: Tuple[str] = (
        "left_eef_pos",
        "left_eef_quat",
        "right_eef_pos",
        "right_eef_quat",
        "hand_joint_state",
    )
    image_keys: Tuple[str] = ("robot_pov_cam",)
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
    run_name: str = "bc"
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
# Main Trainer Class (Simplified with TensorDict)
# ============================================================================


class FlowBCTrainer:
    """Clean trainer class using TensorDict extensively for simplified data handling."""

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
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)

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

        # Setup environment for play
        self._setup_environment()

        # Resume from checkpoint if specified
        if args.resume:
            self._load_checkpoint(args.resume)

        self.step_counter = 0

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

        # Sample data info using TensorDict
        sample = self.dataset[0]
        batch_sample = self._prepare_single_sample(sample)
        tensor_dict = self.dataset.process_batch(batch_sample, self.device)

        print("\n📏 Data Shapes & Types:")
        if tensor_dict["state_seq"] is not None:
            state_seq = tensor_dict["state_seq"]
            print(
                f"  🔢 State: {list(state_seq.shape)} | {state_seq.dtype} | {state_seq.device}"
            )
        if tensor_dict["image_seq"] is not None:
            image_seq = tensor_dict["image_seq"]
            print(
                f"  🖼️  Image: {list(image_seq.shape)} | {image_seq.dtype} | {image_seq.device}"
            )
        actions = tensor_dict["actions"]
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
            print(f"\n🔑 State keys (exact order): {self.dataset.state_keys}")
        if self.args.modality_type in ("image", "state+image"):
            print(f"🖼️  Image keys: {self.dataset.image_keys}")

    def _prepare_single_sample(self, sample: Dict) -> Dict:
        """Prepare a single sample for batch processing."""
        if isinstance(sample.get("obs"), dict):
            # Handle robomimic format
            obs_batch = {}
            for key, value in sample["obs"].items():
                obs_batch[key] = torch.from_numpy(np.array([value])).to(self.device)
            return {
                "obs": obs_batch,
                "actions": torch.from_numpy(np.array([sample["actions"]])).to(
                    self.device
                ),
            }
        else:
            # Handle other formats
            return {
                k: torch.from_numpy(np.array([v])) if isinstance(v, np.ndarray) else v
                for k, v in sample.items()
            }

    def _print_model_info(self):
        """Print comprehensive model information."""
        print_section_header("🤖 MODEL INFORMATION")

        # Model architecture
        print(f"🏗️  Model: {self.model.__class__.__name__}")
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
        # Default keys (maintains exact order as specified)
        state_keys = self.args.state_keys or [
            "left_eef_pos",
            "left_eef_quat",
            "right_eef_pos",
            "right_eef_quat",
            "hand_joint_state",
        ]
        image_keys = self.args.image_keys or ["robot_pov_cam"]

        dataset = SequenceDataset(
            hdf5_path=self.args.dataset,
            state_keys=state_keys,
            image_keys=image_keys,
            sequence_length=self.args.sequence_length,
            frame_stack=self.args.frame_stack,
            pad_sequence=self.args.pad_sequence,
            pad_frame_stack=self.args.pad_frame_stack,
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
            batch_sample = self._prepare_single_sample(sample)
            tensor_dict = self.dataset.process_batch(batch_sample, self.device)

            action_dim = tensor_dict["actions"].shape[-1]
            state_dim = (
                tensor_dict["state_seq"].shape[-1]
                if tensor_dict["state_seq"] is not None
                else 0
            )

            # Update img_size from actual data
            if tensor_dict["image_seq"] is not None:
                self.args.img_size = tensor_dict["image_seq"].shape[-3:]

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
            model = BCPolicy(state_dim, action_dim, self.args).to(self.device)
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

    def _log_best_checkpoint_artifact(self):
        """Log the best checkpoint as a WandB artifact at the end of training."""
        if not self.run or not self.args.validation:
            if not self.args.validation:
                print("⚠️  Best checkpoint not saved because validation is disabled")
            return

        best_path = os.path.join(self.ckpt_dir, "best.pt")
        if os.path.exists(best_path):
            print("🏆 Logging best checkpoint as artifact...")
            artifact = wandb.Artifact(
                name="best_checkpoint",
                type="model",
                description=f"Best model checkpoint with validation loss: {self.best_val_loss:.4f}",
            )
            artifact.add_file(best_path)
            self.run.log_artifact(artifact)
            print("✅ Best checkpoint logged as artifact")
        else:
            print("⚠️  Best checkpoint not found, skipping artifact logging")

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
        """Single training step with flow matching loss using TensorDict."""
        self.model.train()

        # Process batch into TensorDict - dataset ensures state concatenation order
        tensor_dict = self.dataset.process_batch(batch, self.device)

        # Forward pass with mixed precision
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            # Flow matching loss - model receives exact state concatenation order
            flow_loss = self.model.compute_loss(
                tensor_dict["state_seq"],
                tensor_dict["image_seq"],
                tensor_dict["actions"],
            )

            # Sample actions for metrics (no gradients)
            with torch.no_grad():
                pred_actions = self.model.sample_actions(
                    tensor_dict["state_seq"],
                    tensor_dict["image_seq"],
                    steps=self.args.flow_steps,
                )

        # Backward pass
        self.scaler.scale(flow_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(pred_actions, tensor_dict["actions"])
            random_metrics = compute_random_baseline(tensor_dict["actions"])

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
        """Single validation step with action sampling using TensorDict."""
        self.model.eval()

        # Process batch into TensorDict - dataset ensures state concatenation order
        tensor_dict = self.dataset.process_batch(batch, self.device)

        with torch.amp.autocast(
            device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            # Sample actions (main validation metric) - model receives exact state concatenation order
            pred_actions = self.model.sample_actions(
                tensor_dict["state_seq"],
                tensor_dict["image_seq"],
                steps=self.args.val_flow_steps,
            )

        # Compute metrics
        metrics = compute_metrics(pred_actions, tensor_dict["actions"])
        random_metrics = compute_random_baseline(tensor_dict["actions"])

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
            self._log_image_grid(epoch, "train_epoch", self.step_counter)

        # Lightning-style progress bar with better info
        # Get terminal width and set appropriate ncols
        terminal_width = shutil.get_terminal_size().columns
        progress_width = min(terminal_width - 10, 100)  # Leave some margin

        pbar = tqdm(
            self.train_loader,
            desc="  🔥 Training",
            leave=False,
            ncols=progress_width,  # Reduced from 140
            position=1,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",  # Removed rate_fmt
        )

        running_loss = 0.0
        running_mse = 0.0
        running_cos_sim = 0.0
        # Update running averages with more recent bias
        alpha = 0.1  # smoothing factor

        for step, batch in enumerate(pbar):
            # Training step
            step_metrics = self.train_step(batch)
            epoch_metrics.append(step_metrics)

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
                self.run.log(step_metrics, step=self.step_counter)
            self.step_counter += 1

            # Update progress bar with key metrics (simplified)
            pbar.set_postfix(
                {
                    "loss": f"{running_loss:.3f}",  # Reduced precision
                    "mse": f"{running_mse:.3f}",
                    "cos_sim": f"{running_cos_sim:.3f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.0e}",  # Shorter format
                    "gn": f"{get_grad_norm(self.model):.2f}",
                }
            )

        # Epoch-wise averaging (PyTorch Lightning style)
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key.replace("step", "epoch")] = np.mean(
                [m[key] for m in epoch_metrics]
            )

        return avg_metrics

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
            self._log_image_grid(epoch, "val_epoch", self.step_counter)

        # Get terminal width and set appropriate ncols
        terminal_width = shutil.get_terminal_size().columns
        progress_width = min(terminal_width - 10, 100)  # Leave some margin

        pbar = tqdm(
            self.val_loader,
            desc="🔍 Validating",
            leave=False,
            ncols=progress_width,
            position=1,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

        running_mse = 0.0
        running_cos_sim = 0.0
        alpha = 0.1

        for step, batch in enumerate(pbar):
            # Validation step
            step_metrics = self.val_step(batch)
            epoch_metrics.append(step_metrics)

            # Update running averages
            running_mse = (1 - alpha) * running_mse + alpha * step_metrics[
                "val_step/pred_mse"
            ]
            running_cos_sim = (1 - alpha) * running_cos_sim + alpha * step_metrics[
                "val_step/pred_cosine_sim"
            ]

            if self.run:
                self.run.log(step_metrics, step=self.step_counter)
            self.step_counter += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "mse": f"{running_mse:.4f}",
                    "cos_sim": f"{running_cos_sim:.3f}",
                }
            )

        # Epoch-wise averaging
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key.replace("step", "epoch")] = np.mean(
                [m[key] for m in epoch_metrics]
            )

        return avg_metrics

    def _log_image_grid(self, epoch: int, split: str, step: int):
        """Log image grid for visualization using TensorDict."""
        if self.args.modality_type not in ("image", "state+image"):
            return

        # Get a batch
        loader = self.train_loader if split.startswith("train") else self.val_loader
        batch = next(iter(loader))

        # Process batch into TensorDict
        tensor_dict = self.dataset.process_batch(batch, self.device)

        if tensor_dict["image_seq"] is None:
            return

        image_seq = tensor_dict["image_seq"]

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
                step=step,
            )

        print(
            f"✅ Logged {split} image grid: {num_samples} samples × {selected_images.size(1)} timesteps"
        )

    def _init_sequence_buffers(self, initial_obs) -> TensorDict:
        """Initialize rolling sequence buffers using TensorDict for cleaner buffer management."""
        batch_size = self.args.num_eval_envs
        seq_len = self.args.sequence_length * self.args.frame_stack

        def init_state_buffer():
            state_parts = [
                initial_obs[key].view(batch_size, -1)
                for key in self.dataset.state_keys
                if key in initial_obs
            ]
            if state_parts:
                current_state = torch.cat(state_parts, dim=-1)
                return (
                    current_state.unsqueeze(1)
                    .expand(batch_size, seq_len, -1)
                    .contiguous()
                    .to(self.device)
                )
            return None

        def init_image_buffer():
            for key in self.dataset.image_keys:
                if key in initial_obs and initial_obs[key].dim() == 4:
                    img_data = process_image_batch(
                        initial_obs[key], target_format="BCHW", normalize_to_01=True
                    )
                    return (
                        img_data.unsqueeze(1)
                        .expand(batch_size, seq_len, -1, -1, -1)
                        .contiguous()
                        .to(self.device)
                    )
            return None

        buffer_dict = {}

        # Try to initialize buffers, fall back to zeros on any error
        try:
            if self.args.modality_type in ("state", "state+image"):
                buffer_dict["state_seq"] = init_state_buffer() or torch.zeros(
                    batch_size, seq_len, 36, device=self.device
                )

            if self.args.modality_type in ("image", "state+image"):
                C, H, W = self.args.img_size
                buffer_dict["image_seq"] = init_image_buffer() or torch.zeros(
                    batch_size, seq_len, C, H, W, device=self.device
                )
        except Exception as e:
            print(f"⚠️  Error initializing buffers: {e}, using fallback")

        return TensorDict(buffer_dict, batch_size=[batch_size])

    def _update_sequence_buffers(self, obs, buffer_td: TensorDict) -> TensorDict:
        """Update rolling sequence buffers using TensorDict for cleaner buffer management."""
        updated_dict = {}

        # Update state buffer
        if "state_seq" in buffer_td and self.args.modality_type in (
            "state",
            "state+image",
        ):
            state_parts = [
                obs[key].view(obs[key].shape[0], -1)
                for key in self.dataset.state_keys
                if key in obs
            ]
            if state_parts:
                current_state = torch.cat(state_parts, dim=-1).to(self.device)
                state_buffer = torch.roll(buffer_td["state_seq"], shifts=-1, dims=1)
                state_buffer[:, -1, :] = current_state
                updated_dict["state_seq"] = state_buffer
            else:
                updated_dict["state_seq"] = buffer_td["state_seq"]
        else:
            updated_dict["state_seq"] = buffer_td.get("state_seq")

        # Update image buffer
        if "image_seq" in buffer_td and self.args.modality_type in (
            "image",
            "state+image",
        ):
            current_image = None
            for key in self.dataset.image_keys:
                if key in obs and obs[key].dim() == 4:
                    current_image = process_image_batch(
                        obs[key], target_format="BCHW", normalize_to_01=True
                    )
                    break

            if current_image is not None:
                image_buffer = torch.roll(buffer_td["image_seq"], shifts=-1, dims=1)
                image_buffer[:, -1, :, :, :] = current_image.to(self.device)
                updated_dict["image_seq"] = image_buffer
            else:
                updated_dict["image_seq"] = buffer_td["image_seq"]
        else:
            updated_dict["image_seq"] = buffer_td.get("image_seq")

        return TensorDict(updated_dict, batch_size=buffer_td.batch_size)

    def _handle_episode_rewards(
        self,
        done: torch.Tensor,
        current_rewards: torch.Tensor,
        episode_rewards: List,
        buffer_td: TensorDict,
    ) -> torch.Tensor:
        """Handle episode completion and reset buffers."""
        for i in range(self.args.num_eval_envs):
            if done[i]:
                if len(episode_rewards) < 100:  # Limit logging
                    episode_rewards.append(current_rewards[i].item())
                current_rewards[i] = 0.0

                # Reset sequence buffers for this environment using TensorDict
                if "state_seq" in buffer_td:
                    buffer_td["state_seq"][i] = 0.0
                if "image_seq" in buffer_td:
                    buffer_td["image_seq"][i] = 0.0
        return current_rewards

    @torch.no_grad()
    def play_in_environment(self, epoch: int):
        """Play model in IsaacLab environment using TensorDict for buffer management."""
        if self.eval_envs is None:
            print("⚠️  Environment not available, skipping play")
            return

        print(f"🎮 Playing model in environment (epoch {epoch})")
        self.model.eval()

        # Initialize environment and tracking variables
        obs, _ = self.eval_envs.reset()
        step_count = 0
        done = torch.zeros(self.args.num_eval_envs, dtype=torch.bool)
        episode_rewards = []
        current_rewards = torch.zeros(self.args.num_eval_envs)

        # Initialize rolling buffers using TensorDict
        buffer_td = self._init_sequence_buffers(obs)

        play_pbar = tqdm(
            range(self.args.num_eval_steps), desc="🎮 Playing", leave=False, ncols=100
        )

        for step in play_pbar:
            if done.all():
                break

            # Update rolling buffers and get actions
            buffer_td = self._update_sequence_buffers(obs, buffer_td)

            with torch.amp.autocast(
                device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
            ):
                # Extract actual tensors from TensorDict
                state_seq = buffer_td["state_seq"] if "state_seq" in buffer_td else None
                image_seq = buffer_td["image_seq"] if "image_seq" in buffer_td else None

                pred_actions = self.model.sample_actions(
                    state_seq,
                    image_seq,
                    steps=self.args.val_flow_steps,
                    deterministic=True,
                )

            # Process actions and step environment
            if self.action_normalizer:
                pred_actions = self.action_normalizer.denormalize(pred_actions)
            obs, rewards, done, _ = self.eval_envs.step(pred_actions.float())

            # Track and handle rewards/episodes
            current_rewards += rewards.cpu()
            current_rewards = self._handle_episode_rewards(
                done, current_rewards, episode_rewards, buffer_td
            )

            step_count += 1
            play_pbar.set_postfix(
                {
                    "completed": f"{done.sum().item()}/{self.args.num_eval_envs}",
                    "avg_reward": f"{np.mean(episode_rewards):.2f}"
                    if episode_rewards
                    else "N/A",
                }
            )

        self.eval_envs.reset()
        # Log results
        if episode_rewards and self.run:
            self.run.log(
                {
                    "play/avg_reward": np.mean(episode_rewards),
                    "play/max_reward": np.max(episode_rewards),
                    "play/min_reward": np.min(episode_rewards),
                    "play/episodes_completed": len(episode_rewards),
                    "play/steps_taken": step_count,
                },
                step=self.step_counter,
            )
            self.step_counter += 1

        print(
            f"🎯 Play completed: {len(episode_rewards)} episodes, avg reward: {np.mean(episode_rewards):.2f}"
        )

    def _handle_checkpoints_and_validation(self, epoch: int, val_metrics: Dict) -> bool:
        """Handle validation, checkpoints, and best model tracking."""
        is_best = False

        if self.args.validation and epoch % self.args.eval_freq == 0:
            val_metrics.update(self.val_epoch(epoch))
            val_loss = val_metrics["val_epoch/pred_mse"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

        # Save regular checkpoint
        if epoch % self.args.save_freq == 0:
            ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
            self._save_checkpoint(ckpt_path, epoch, is_best=False)

        # Save best checkpoint
        if self.args.validation and is_best:
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            self._save_checkpoint(best_path, epoch, is_best=True)

        return is_best

    def _handle_environment_play(self, epoch: int):
        """Handle environment play scheduling."""
        if self.args.validation:
            # When validation is enabled, play less frequently after validation
            if epoch % (self.args.eval_freq * 2) == 0:
                self.play_in_environment(epoch)
        else:
            # When validation is disabled, play after every eval_freq epochs
            if epoch % self.args.eval_freq == 0:
                self.play_in_environment(epoch)

    def _log_epoch_summary(
        self,
        epoch: int,
        epoch_metrics: Dict,
        is_best: bool,
        old_lr: float,
        new_lr: float,
    ):
        """Log epoch summary and metrics."""
        if self.run:
            self.run.log(epoch_metrics, step=self.step_counter)

        # Print epoch summary (Lightning style)
        metrics_str = []
        for k, v in epoch_metrics.items():
            if isinstance(v, float) and "time" not in k and k != "learning_rate":
                metrics_str.append(f"{k}: {v:.4f}")

        status_emoji = "🏆" if is_best else "✅"
        lr_info = f"lr: {new_lr:.1e}" if new_lr != old_lr else ""
        validation_info = " [NO VAL]" if not self.args.validation else ""

        print(
            f"{status_emoji} Epoch {epoch:3d}{validation_info} | "
            + " | ".join(metrics_str)
            + (f" | 📈 {lr_info}" if lr_info else "")
            + f" | ⏱️  {epoch_metrics['epoch_time']:.1f}s"
        )

    def fit(self):
        """Main training loop with comprehensive logging."""
        print_section_header("🚀 TRAINING STARTED")
        print(f"📅 Epochs: {self.start_epoch + 1} → {self.args.num_epochs}")
        print(f"💾 Checkpoint dir: {self.ckpt_dir}")
        print(f"📊 WandB: {'Online' if self.args.log else 'Offline'}")
        print(f"🔬 Validation: {'Enabled' if self.args.validation else 'Disabled'}")

        play_freq = (
            self.args.eval_freq * 2 if self.args.validation else self.args.eval_freq
        )
        print(f"🎮 Environment play: Every {play_freq} epochs")

        for epoch in range(self.start_epoch + 1, self.args.num_epochs + 1):
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation, checkpoints, and best model tracking
            val_metrics = {}
            is_best = self._handle_checkpoints_and_validation(epoch, val_metrics)

            # Environment play
            self._handle_environment_play(epoch)

            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]

            if old_lr != new_lr:
                print(f"📉 Learning rate decayed: {old_lr:.2e} → {new_lr:.2e}")

            # Prepare and log epoch metrics
            epoch_metrics = {
                **train_metrics,
                **val_metrics,
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "epoch_time": time.time() - epoch_start,
                "learning_rate": new_lr,
            }

            self._log_epoch_summary(epoch, epoch_metrics, is_best, old_lr, new_lr)

        # Cleanup
        if self.args.validation:
            self._log_best_checkpoint_artifact()

        if self.run:
            self.run.finish()

        print_section_header("🎉 TRAINING COMPLETED")

    def __del__(self):
        """Cleanup environment on deletion."""
        if hasattr(self, "eval_envs") and self.eval_envs is not None:
            try:
                self.eval_envs.close()
            except Exception as e:
                print(f"❌ Error closing evaluation environment: {e}")
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
