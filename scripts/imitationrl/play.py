"""Play VanillaFlow-BC policy in the Isaac Lab simulation.

Loads a VanillaFlowPolicy checkpoint (from scripts/imitationrl/train.py),
recreates the policy, and runs it in an Isaac Lab environment. It builds a
rolling observation sequence (state + image) and feeds it to the policy at
each step.

Notes
- Assumes the env's policy observation is [flattened image | flattened state].
  If not available, it falls back to zeros for the missing part.
- Action de-normalization uses `action_normalizer` saved in the checkpoint when
  `normalize_actions=True` during training; otherwise actions are forwarded.
- Initializes an OFFLINE W&B run and routes RecordVideo outputs under the run's files dir.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image

from scripts.imitationrl.models import VanillaFlowPolicy  # Changed from SeqFlowPolicy
from scripts.imitationrl.utils import process_image_batch
from scripts.utils import (
    load_args,
    make_isaaclab_env,
    seed_everything,
)

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

WANDB_DIR_DEFAULT = "/home/chandramouli/cognitiverl"


@dataclass
class EnvArgs:
    # Isaac Lab task + runtime
    task: str = "Isaac-NutPour-GR1T2-Pink-IK-Abs-v0"
    device: str = "cuda:0"
    seed: int = 1
    num_envs: int = 32
    num_steps: int = 350
    capture_video: bool = False
    disable_fabric: bool = False
    video_length: int = 350
    video_interval: int = 2000
    headless: bool = False
    enable_cameras: bool = False


def save_image_buffer(
    image_buffer: torch.Tensor,
    save_dir: str,
    step: int,
    env_indices: Optional[List[int]] = None,
    max_envs_to_save: int = 4,
):
    """Save image buffer as a single grid: envs as rows, sequence timesteps as columns.

    Args:
        image_buffer: (num_envs, T, C, H, W) tensor
        save_dir: Directory to save images
        step: Current step number
        env_indices: Which environments to save (if None, save first max_envs_to_save)
        max_envs_to_save: Maximum number of environments to save
    """
    if image_buffer is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    num_envs, T, C, H, W = image_buffer.shape

    # Select which environments to save
    if env_indices is None:
        num_envs_to_save = min(max_envs_to_save, num_envs)
        env_indices = list(range(num_envs_to_save))
    else:
        num_envs_to_save = len(env_indices)

    # Get selected environments: (num_envs_to_save, T, C, H, W)
    selected_envs = image_buffer[env_indices]

    # Ensure values are in [0, 1] range
    if selected_envs.max() > 1.0:
        selected_envs = selected_envs / 255.0
    selected_envs = torch.clamp(selected_envs, 0.0, 1.0)

    # Reshape for grid: (num_envs_to_save * T, C, H, W)
    # We want envs as rows, timesteps as columns
    grid_images = selected_envs.view(num_envs_to_save * T, C, H, W)

    # Create grid: T columns (timesteps), num_envs_to_save rows (environments)
    grid = torchvision.utils.make_grid(
        grid_images,
        nrow=T,  # T timesteps per row (each row is one environment)
        normalize=False,  # Already normalized
        padding=2,
        pad_value=1.0,  # White padding
    )

    # Convert to PIL image
    grid_np = grid.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    grid_np = (grid_np * 255).astype(np.uint8)

    # Save single grid image
    filename = f"step_{step:04d}_grid_{num_envs_to_save}envs_{T}timesteps.png"
    filepath = os.path.join(save_dir, filename)

    Image.fromarray(grid_np).save(filepath)

    # Also save a version with labels
    save_labeled_grid(
        grid_np,
        filepath.replace(".png", "_labeled.png"),
        num_envs_to_save,
        T,
        env_indices,
    )


def save_labeled_grid(
    grid_np: np.ndarray,
    filepath: str,
    num_envs: int,
    num_timesteps: int,
    env_indices: List[int],
):
    """Save grid with environment and timestep labels."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(grid_np)
        draw = ImageDraw.Draw(img)

        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
                )
            except:
                font = ImageFont.load_default()

        # Calculate image dimensions in the grid
        img_height = grid_np.shape[0] // num_envs
        img_width = grid_np.shape[1] // num_timesteps

        # Add environment labels (left side)
        for env_idx in range(num_envs):
            y_pos = env_idx * img_height + img_height // 2
            label = f"Env {env_indices[env_idx]}"
            draw.text((5, y_pos), label, fill=(255, 0, 0), font=font)

        # Add timestep labels (top)
        for t_idx in range(num_timesteps):
            x_pos = t_idx * img_width + img_width // 2 - 20
            label = f"t-{num_timesteps - 1 - t_idx}"  # Most recent is t-0
            draw.text((x_pos, 5), label, fill=(0, 255, 0), font=font)

        img.save(filepath)

    except Exception:
        # If labeling fails, just save the original
        Image.fromarray(grid_np).save(filepath)


def save_obs_grid(
    image_buffer: torch.Tensor,
    save_path: str,
    max_envs_to_save: int = 8,
    max_timesteps: int = 10,
) -> None:
    """Save observation grid showing multiple environments and timesteps."""
    if image_buffer is None or image_buffer.numel() == 0:
        print("⚠️  Image buffer is empty, skipping grid save")
        return

    B, T, C, H, W = image_buffer.shape
    num_envs_to_save = min(max_envs_to_save, B)
    T = min(max_timesteps, T)

    # Select random environments to save
    env_indices = torch.randperm(B)[:num_envs_to_save]

    # Get selected environments: (num_envs_to_save, T, C, H, W)
    selected_envs = image_buffer[env_indices]

    # Use shared image processing function
    selected_envs = process_image_batch(
        selected_envs, target_format="BTCHW", normalize_to_01=True
    )

    # Reshape for grid: (num_envs_to_save * T, C, H, W)
    # We want envs as rows, timesteps as columns
    grid_images = selected_envs.view(num_envs_to_save * T, C, H, W)

    # Create grid: T columns (timesteps), num_envs_to_save rows
    grid = torchvision.utils.make_grid(
        grid_images,
        nrow=T,  # T timesteps per row
        normalize=False,  # Already normalized
        padding=2,
        pad_value=0.5,  # Gray padding
    )

    # Convert to PIL and save
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(save_path)
    print(f"💾 Saved observation grid ({num_envs_to_save}×{T}) to {save_path}")


def _save_obs_for_envs(
    obs_dict: Dict,
    adapter: "ObsAdapter",
    save_dir: str,
    step: int,
    max_envs_to_save: int = 8,
) -> None:
    """Save observations from selected environments."""
    if adapter.image_key not in obs_dict:
        return

    try:
        os.makedirs(save_dir, exist_ok=True)

        # Get image observations
        img_obs = obs_dict[adapter.image_key]  # (num_envs, H, W, C)

        if isinstance(img_obs, torch.Tensor):
            # Use shared image processing function
            img_obs = process_image_batch(
                img_obs, target_format="BCHW", normalize_to_01=True
            )

            # Select environments to save
            num_envs_to_save = min(max_envs_to_save, img_obs.shape[0])
            env_indices = torch.randperm(img_obs.shape[0])[:num_envs_to_save]
            selected_obs = img_obs[env_indices]  # (num_envs_to_save, C, H, W)

            # Save each environment's observation
            for i, env_idx in enumerate(env_indices):
                # Convert to numpy and save
                obs_np = (selected_obs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                )
                save_path = os.path.join(
                    save_dir, f"step{step:04d}_env{env_idx:02d}.png"
                )
                Image.fromarray(obs_np).save(save_path)

            print(
                f"💾 Saved {num_envs_to_save} environment observations at step {step}"
            )

    except Exception as e:
        print(f"⚠️  Error saving observations: {e}")


@dataclass
class PlayArgs(EnvArgs):
    # Checkpoint + AMP
    checkpoint: str = "checkpoints/best.pt"
    amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" or "fp16"
    val_flow_steps: int = 32
    # Logging (offline wandb)
    wandb_project: str = "imitation-play"
    run_name: str = "vanilla_flow_play"  # Updated name
    wandb_dir: str = WANDB_DIR_DEFAULT
    log_interval: int = 10

    # Image saving options - ADD THESE
    save_observation_images: bool = True
    save_image_interval: int = 1  # Save every N steps
    max_envs_to_save: int = 4  # How many environments to save images for
    save_sequence_buffer: bool = True  # Save the full sequence buffer
    save_current_obs: bool = True  # Save current observation


class ObsAdapter:
    """Minimal observation adapter: dict -> (state[N,S], image[N,C,H,W])."""

    def __init__(
        self,
        state_keys: List[str],
        image_key: Optional[str],
        img_size: Tuple[int, int, int],
    ):
        self.state_keys = state_keys
        self.image_key = image_key
        self.C, self.H, self.W = img_size

    def _to_tensor(self, x) -> torch.Tensor:
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x)

    def _ensure_float(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype.is_floating_point:
            return x.float()
        return x.float()

    def from_obs(self, obs_in: Dict) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Unwrap policy dict if present
        obs = obs_in.get("policy", obs_in)

        # Extract state
        state_parts = []
        for key in self.state_keys:
            if key in obs:
                val = self._to_tensor(obs[key])
                state_parts.append(val.view(val.shape[0], -1))

        if state_parts:
            state = torch.cat(state_parts, dim=-1)
        else:
            # Fallback to zeros if no state keys found
            state = torch.zeros(1, self.S, device=self.device)

        # Extract image
        image = None
        if self.image_key and self.image_key in obs:
            img = self._to_tensor(obs[self.image_key])  # likely (N,H,W,C) or (N,C,H,W)

            # Use shared image processing function
            image = process_image_batch(img, target_format="BCHW", normalize_to_01=True)

            # Ensure expected channels/size if possible
            if image.shape[1] != self.C:
                # try to adapt by slicing or repeating
                if image.shape[1] > self.C:
                    image = image[:, : self.C]
                else:
                    repeat = (self.C + image.shape[1] - 1) // image.shape[1]
                    image = image.repeat(1, repeat, 1, 1)[:, : self.C]

        return state.float(), image.float() if image is not None else None


def _infer_dims_from_state_dict(sd: dict) -> Tuple[int, int, dict]:
    """Infer (state_dim, action_dim, model_params) from the VanillaFlowPolicy state dict."""

    # action_dim from final flow layer
    action_dim = None
    ff_hidden = None
    for k in sd.keys():
        if "flow.6.weight" in k:  # Final layer
            action_dim = sd[k].shape[0]
            print(f"Found action_dim={action_dim} from {k}")
        elif "flow.0.weight" in k:  # First layer
            ff_hidden = sd[k].shape[0]
            flow_input_dim = sd[k].shape[1]
            print(
                f"Found ff_hidden={ff_hidden}, flow_input_dim={flow_input_dim} from {k}"
            )

    if action_dim is None or ff_hidden is None:
        raise RuntimeError("Could not infer action_dim and ff_hidden from state dict")

    # d_model from flow input: flow_input = d_model + action_dim + 1
    d_model = flow_input_dim - action_dim - 1
    print(f"Inferred d_model={d_model}")

    # state_dim from state encoder if available
    state_dim = 0
    for k in ("state_encoder.net.0.weight",):
        if k in sd:
            state_dim = sd[k].shape[1]
            print(f"Found state_dim={state_dim} from {k}")
            break

    if state_dim == 0:
        state_dim = 36  # Fallback
        print(f"Using fallback state_dim={state_dim}")

    # Try to infer frame_stack from sequence_processor
    frame_stack = 10  # default
    sequence_length = 1  # default

    for k in sd.keys():
        if "sequence_processor.net.0.weight" in k:
            effective_seq_length = frame_stack * sequence_length
            seq_input_dim = effective_seq_length * d_model
            print(
                f"Inferred frame_stack={frame_stack} from sequence processor input dim {seq_input_dim}"
            )
            break

    model_params = {
        "d_model": d_model,
        "ff_hidden": ff_hidden,
        "frame_stack": frame_stack,
        "action_dim": action_dim,
        "state_dim": state_dim,
    }

    return int(state_dim), int(action_dim), model_params


class SequenceBuffer:
    def __init__(
        self, T: int, num_envs: int, state_dim: int, img_size: Tuple[int, int, int]
    ):
        self.T = T
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.C, self.H, self.W = img_size
        self._state = torch.zeros(num_envs, T, state_dim, dtype=torch.float32)
        self._image = torch.zeros(
            num_envs, T, self.C, self.H, self.W, dtype=torch.float32
        )

    def to(self, device: torch.device):
        self._state = self._state.to(device, non_blocking=True)
        self._image = self._image.to(device, non_blocking=True)
        return self

    def append(self, state: torch.Tensor, image: Optional[torch.Tensor]):
        # state: (N, S) ; image: (N, C, H, W) or None
        self._state = torch.roll(self._state, shifts=-1, dims=1)
        self._state[:, -1, :] = state
        self._image = torch.roll(self._image, shifts=-1, dims=1)
        if image is not None:
            self._image[:, -1, :, :, :] = image
        else:
            self._image[:, -1, :, :, :] = 0.0

    def reset_indices(self, done_mask: torch.Tensor):
        # done_mask: (N,) bool
        if done_mask is None or done_mask.numel() == 0:
            return
        idx = done_mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return
        self._state[idx, :, :] = 0.0
        self._image[idx, :, :, :, :] = 0.0

    @property
    def state(self) -> torch.Tensor:
        return self._state

    @property
    def image(self) -> torch.Tensor:
        return self._image


def main():
    args = load_args(PlayArgs)
    seed_everything(args.seed, use_torch=True, torch_deterministic=True)

    # Load checkpoint and reconstruct policy config
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_sd = ckpt["model_state_dict"]

    # Updated to match train.py action normalizer format
    action_normalizer = None
    if "action_normalizer" in ckpt and ckpt["action_normalizer"]:
        from scripts.imitationrl.dataset import ActionNormalizer

        action_normalizer = ActionNormalizer.from_stats(ckpt["action_normalizer"])

    # Updated dimension inference
    state_dim_tr, action_dim, inferred_params = _infer_dims_from_state_dict(model_sd)

    # Build model args namespace from checkpoint - Updated for VanillaFlowPolicy
    required_model_keys = [
        "d_model",
        "nhead",
        "num_layers",
        "ff_hidden",
        "dropout",
        "sequence_length",
        "img_size",
        "modality_type",
        "frame_stack",
    ]

    # Use inferred parameters first, then checkpoint args, then defaults
    model_args_dict = {}
    for k in required_model_keys:
        if k in inferred_params:
            model_args_dict[k] = inferred_params[k]
            print(f"Using inferred {k}={inferred_params[k]}")
        elif k in ckpt_args:
            model_args_dict[k] = ckpt_args[k]
            print(f"Using checkpoint {k}={ckpt_args[k]}")
        else:
            # Provide defaults matching train.py
            defaults = {
                "d_model": 256,
                "nhead": 8,
                "num_layers": 4,
                "ff_hidden": 512,
                "dropout": 0.1,
                "sequence_length": 1,
                "img_size": (3, 160, 256),
                "modality_type": "state+image",
                "frame_stack": 10,
            }
            if k in defaults:
                model_args_dict[k] = defaults[k]
                print(f"Using default {k}={defaults[k]}")

    model_args = SimpleNamespace(**model_args_dict)

    # Device + AMP
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and bool(args.amp)
    amp_dtype = (
        torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
    )

    # Initialize offline W&B (for metrics + to route videos under run dir)
    run = None
    files_dir = None
    if wandb is not None:
        try:
            os.makedirs(args.wandb_dir, exist_ok=True)
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = args.wandb_dir
            run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
                dir=args.wandb_dir,
                settings=wandb.Settings(code_dir=os.getcwd()),
                save_code=False,
            )
            base_dir = run.dir if run is not None else args.wandb_dir
            files_dir = (
                base_dir
                if os.path.basename(base_dir) == "files"
                else os.path.join(base_dir, "files")
            )
            os.makedirs(files_dir, exist_ok=True)
        except Exception:
            run = None
            files_dir = None

    # Start Isaac app
    from argparse import Namespace

    import pinocchio  # noqa: F401
    from isaaclab.app import AppLauncher

    app = AppLauncher(Namespace(**asdict(args))).app

    import isaaclab_tasks  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

    # Create environment
    env_fn = make_isaaclab_env(
        task=args.task,
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        capture_video=args.capture_video,
        disable_fabric=args.disable_fabric,
        log_dir=files_dir,
        video_length=args.video_length,
        video_interval=args.video_interval,
    )
    env = env_fn()

    # Build policy - Print model creation details
    print("\n🔧 Creating VanillaFlowPolicy:")
    print(f"  state_dim: {state_dim_tr}")
    print(f"  action_dim: {action_dim}")
    print(f"  d_model: {model_args.d_model}")
    print(f"  ff_hidden: {model_args.ff_hidden}")
    print(f"  frame_stack: {model_args.frame_stack}")
    print(f"  sequence_length: {model_args.sequence_length}")

    policy = VanillaFlowPolicy(state_dim_tr, action_dim, model_args).to(device)

    # Debug: Print some layer shapes before loading
    print("\n🔍 Model layer shapes:")
    for name, param in policy.named_parameters():
        if "flow" in name and "weight" in name:
            print(f"  {name}: {param.shape}")

    policy.load_state_dict(model_sd)
    policy.eval()
    print("✅ Model loaded successfully")

    # Buffers
    T = int(model_args.sequence_length * model_args.frame_stack)
    C, H, W = tuple(model_args.img_size)
    buf = SequenceBuffer(
        T=T, num_envs=args.num_envs, state_dim=state_dim_tr, img_size=(C, H, W)
    ).to(device)

    # Prepare observation adapter using keys from checkpoint if present
    state_keys = ckpt_args.get(
        "state_keys",
        [
            "left_eef_pos",
            "left_eef_quat",
            "right_eef_pos",
            "right_eef_quat",
            "hand_joint_state",
        ],
    )

    # Ensure state_keys is not None
    if state_keys is None:
        state_keys = [
            "left_eef_pos",
            "left_eef_quat",
            "right_eef_pos",
            "right_eef_quat",
            "hand_joint_state",
        ]

    image_keys = ckpt_args.get("image_keys", ("robot_pov_cam",))

    # Ensure image_keys is not None and handle different formats
    if image_keys is None:
        image_keys = ["robot_pov_cam"]

    # Handle different image_keys formats (list, tuple, string)
    if isinstance(image_keys, str):
        image_key = image_keys
    elif isinstance(image_keys, (list, tuple)) and len(image_keys) > 0:
        image_key = image_keys[0]
    else:
        image_key = "robot_pov_cam"  # fallback

    print(f"🔑 Using state_keys: {state_keys}")
    print(f"🖼️  Using image_key: {image_key}")

    adapter = ObsAdapter(list(state_keys), image_key, (C, H, W))

    # Reset env and prefill buffer with initial obs
    obs_dict, _ = env.reset()
    s0, i0 = adapter.from_obs(obs_dict)
    s0 = s0.to(device)
    i0 = i0.to(device) if i0 is not None else None
    # Fill T times
    for _ in range(T):
        buf.append(s0, i0)

    # Action de-normalization helper - Updated to use ActionNormalizer
    def denormalize_actions(x: torch.Tensor) -> torch.Tensor:
        if action_normalizer is not None:
            return action_normalizer.denormalize(x)
        return x

    # Episode/metric trackers
    ep_returns = torch.zeros(args.num_envs, dtype=torch.float32)
    ep_lengths = torch.zeros(args.num_envs, dtype=torch.long)
    completed_returns: List[float] = []
    completed_lengths: List[int] = []
    total_episodes = 0

    # Create directories for saving images
    image_save_dirs = {}
    if args.save_observation_images:
        base_image_dir = os.path.join(files_dir or ".", "videos", "observations")
        image_save_dirs["sequence_buffer"] = os.path.join(
            base_image_dir, "sequence_buffer"
        )
        image_save_dirs["current_obs"] = os.path.join(base_image_dir, "current_obs")

        for dir_path in image_save_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        print(f"🖼️  Saving observation images to: {base_image_dir}")
        print(f"   📁 Sequence buffer: {image_save_dirs['sequence_buffer']}")
        print(f"   📁 Current obs: {image_save_dirs['current_obs']}")

    # Step loop
    for step in range(1, args.num_steps + 1):
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type="cuda", enabled=amp_enabled, dtype=amp_dtype
            ),
        ):
            state_seq = buf.state
            image_seq = buf.image

            # Save sequence buffer images if enabled
            if (
                args.save_observation_images
                and args.save_sequence_buffer
                and step % args.save_image_interval == 0
            ):
                save_image_buffer(
                    image_seq,
                    image_save_dirs["sequence_buffer"],
                    step,
                    max_envs_to_save=args.max_envs_to_save,
                )

            pred = policy.sample_actions(
                state_seq,
                image_seq,
                steps=int(args.val_flow_steps),
                deterministic=True,
            )
            act = denormalize_actions(pred)

        # Clamp to env bounds if available
        if hasattr(env, "action_space"):
            low = torch.as_tensor(env.action_space.low, device=act.device)
            high = torch.as_tensor(env.action_space.high, device=act.device)
            # Broadcast to (N, A)
            if low.ndim == 1:
                low = low.unsqueeze(0).expand_as(act)
                high = high.unsqueeze(0).expand_as(act)
            act = torch.max(torch.min(act, high), low)

        # Step env (Wrapper API): returns obs_dict, reward, done, extras
        obs_dict, rew, done, info = env.step(act)

        # Save current observation images if enabled
        if (
            args.save_observation_images
            and args.save_current_obs
            and step % args.save_image_interval == 0
        ):
            _save_obs_for_envs(
                obs_dict,
                adapter,
                image_save_dirs["current_obs"],
                step,
                max_envs_to_save=args.max_envs_to_save,
            )

        # Accumulate episode returns/lengths
        rew_cpu = (
            rew.detach().to("cpu").view(-1)
            if isinstance(rew, torch.Tensor)
            else torch.tensor(rew, dtype=torch.float32).view(-1)
        )
        done_cpu = (
            done.to(torch.bool).to("cpu").view(-1)
            if isinstance(done, torch.Tensor)
            else torch.as_tensor(done, dtype=torch.bool).view(-1)
        )
        ep_returns[: rew_cpu.numel()] += rew_cpu
        ep_lengths[: done_cpu.numel()] += 1

        s, i = adapter.from_obs(obs_dict)
        s = s.to(device)
        i = i.to(device) if i is not None else None
        buf.append(s, i)

        # Reset buffers for terminated envs
        if isinstance(done, torch.Tensor):
            buf.reset_indices(done.to(torch.bool).to(device))
        else:
            done_t = torch.as_tensor(done, dtype=torch.bool, device=device)
            buf.reset_indices(done_t)

        # For environments that finished, record metrics and reset trackers
        if done_cpu.any():
            finished_idx = torch.nonzero(done_cpu, as_tuple=False).view(-1)
            if finished_idx.numel() > 0:
                completed_returns.extend(ep_returns[finished_idx].tolist())
                completed_lengths.extend(ep_lengths[finished_idx].tolist())
                total_episodes += int(finished_idx.numel())
                ep_returns[finished_idx] = 0.0
                ep_lengths[finished_idx] = 0

        # Periodic logging to offline W&B
        if run is not None and (step % max(1, int(args.log_interval)) == 0):
            log_dict = {
                "reward/step_mean": float(rew_cpu.mean().item())
                if rew_cpu.numel() > 0
                else 0.0,
                "reward/step_min": float(rew_cpu.min().item())
                if rew_cpu.numel() > 0
                else 0.0,
                "reward/step_max": float(rew_cpu.max().item())
                if rew_cpu.numel() > 0
                else 0.0,
                "env/episodes_total": int(total_episodes),
                "env/num_envs": int(args.num_envs),
                "env/step": int(step),
            }
            if len(completed_returns) > 0:
                cr_t = torch.tensor(completed_returns, dtype=torch.float32)
                cl_t = torch.tensor(completed_lengths, dtype=torch.float32)
                log_dict.update(
                    {
                        "episode/return_mean": float(cr_t.mean().item()),
                        "episode/return_min": float(cr_t.min().item()),
                        "episode/return_max": float(cr_t.max().item()),
                        "episode/length_mean": float(cl_t.mean().item())
                        if cl_t.numel() > 0
                        else 0.0,
                    }
                )
                completed_returns.clear()
                completed_lengths.clear()
            try:
                wandb.log(log_dict, step=step)
            except Exception:
                pass

        if step % 50 == 0 or step == args.num_steps:
            print(f"Step {step}/{args.num_steps}")
            if completed_returns:
                print(
                    f"  Avg reward: {sum(completed_returns) / len(completed_returns):.2f}"
                )

            # Print image saving status
            if args.save_observation_images:
                total_images = (
                    step // args.save_image_interval
                ) * args.max_envs_to_save
                print(f"  📸 Saved ~{total_images} observation images")

    # Final summary
    if completed_returns:
        avg_reward = sum(completed_returns) / len(completed_returns)
        print("\n🎯 Final Results:")
        print(f"  Episodes completed: {len(completed_returns)}")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Max reward: {max(completed_returns):.2f}")
        print(f"  Min reward: {min(completed_returns):.2f}")

    if run:
        wandb.finish()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
