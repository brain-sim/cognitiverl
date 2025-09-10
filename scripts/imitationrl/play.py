"""Play SeqFlow-BC policy in the Isaac Lab simulation.

Loads a SeqFlowPolicy checkpoint (from scripts/imitation_learning/train.py),
recreates the policy, and runs it in an Isaac Lab environment. It builds a
rolling observation sequence (state + image) and feeds it to the policy at
each step.

Notes
- Assumes the env's policy observation is [flattened image | flattened state].
  If not available, it falls back to zeros for the missing part.
- Action de-normalization uses `action_stats` saved in the checkpoint when
  `normalize_actions=True` during training; otherwise actions are forwarded.
- Initializes an OFFLINE W&B run and routes RecordVideo outputs under the run's files dir.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

from scripts.imitationrl.model import SeqFlowPolicy
from scripts.utils import load_args, make_isaaclab_env, seed_everything

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


@dataclass
class PlayArgs(EnvArgs):
    # Checkpoint + AMP
    checkpoint: str = "checkpoints/best.pt"
    amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" or "fp16"
    val_flow_steps: int = 32
    # Logging (offline wandb)
    wandb_project: str = "imitation-play"
    run_name: str = "seq_flow_play"
    wandb_dir: str = WANDB_DIR_DEFAULT
    log_interval: int = 10


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

        # Build state by concatenating requested keys if present
        state_parts = []
        for k in self.state_keys:
            if k in obs:
                v = self._to_tensor(obs[k])
                v = v.view(v.shape[0], -1)  # (N, *)
                state_parts.append(v)
        if state_parts:
            state = torch.cat(state_parts, dim=-1)
        else:
            # fallback to zeros if nothing available
            N = next(iter(obs.values())).shape[0]
            state = torch.zeros(N, 0)

        # Build image if key present
        image = None
        if self.image_key and self.image_key in obs:
            img = self._to_tensor(obs[self.image_key])  # likely (N,H,W,C) or (N,C,H,W)
            if img.ndim == 4 and img.shape[-1] in (1, 3, 4):
                # NHWC -> NCHW
                img = img.permute(0, 3, 1, 2)
            image = self._ensure_float(img)
            # If integer or >1.0, scale to [0,1]
            if not image.dtype.is_floating_point or image.max() > 1.0:
                image = image.float()
                if image.max() > 1.0:
                    image = (image / 255.0).clamp(0.0, 1.0)

            # Ensure expected channels/size if possible
            if image.shape[1] != self.C:
                # try to adapt by slicing or repeating
                if image.shape[1] > self.C:
                    image = image[:, : self.C]
                else:
                    repeat = (self.C + image.shape[1] - 1) // image.shape[1]
                    image = image.repeat(1, repeat, 1, 1)[:, : self.C]
        return state.float(), image.float() if image is not None else None


def _infer_dims_from_state_dict(sd: dict) -> Tuple[int, int]:
    """Infer (state_dim, action_dim) from the model state dict."""
    # action_dim from last flow Linear weight
    # Try common key name patterns
    action_dim = None
    for k in ("flow.4.weight", "flow.2.weight", "flow.6.weight"):
        if k in sd:
            action_dim = sd[k].shape[0]
            break
    if action_dim is None:
        raise RuntimeError(
            "Could not infer action_dim from state dict (flow.*.weight not found)"
        )

    # state_dim from first StateMLP layer
    state_dim = 0
    for k in ("state_encoder.net.0.weight", "state_encoder.0.weight"):
        if k in sd:
            state_dim = sd[k].shape[1]
            break
    return int(state_dim), int(action_dim)


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


def _split_env_obs(
    obs: torch.Tensor, state_dim: int, img_size: Tuple[int, int, int]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split env policy observation into (state, image).

    Heuristic: assumes obs is [flattened image | flattened state | ...]. If not
    enough dims, falls back to zeros for the missing part.
    """
    N = obs.size(0)
    C, H, W = img_size
    img_flat = C * H * W
    D = obs.size(1)
    state_out = torch.zeros(N, state_dim, device=obs.device, dtype=torch.float32)
    img_out: Optional[torch.Tensor] = None

    # Image slice if available
    if D >= img_flat:
        img = obs[:, :img_flat].view(N, C, H, W)
        # Normalize pixel range if needed (env may already be [0,1])
        img_out = img.to(torch.float32)
        off = img_flat
    else:
        off = 0

    # State slice if available
    if D - off >= state_dim and state_dim > 0:
        state_out = obs[:, off : off + state_dim].to(torch.float32)
    elif state_dim > 0 and D > off:
        # Partial fill if obs smaller than expected
        take = min(state_dim, D - off)
        state_out[:, :take] = obs[:, off : off + take].to(torch.float32)

    return state_out, img_out


def main():
    args = load_args(PlayArgs)
    seed_everything(args.seed, use_torch=True, torch_deterministic=True)

    # Load checkpoint and reconstruct policy config
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_sd = ckpt["model_state_dict"]
    action_stats = ckpt.get("action_stats", None)

    state_dim_tr, action_dim = _infer_dims_from_state_dict(model_sd)

    # Build model args namespace from checkpoint (must provide architecture fields)
    required_model_keys = [
        "d_model",
        "nhead",
        "num_layers",
        "ff_hidden",
        "dropout",
        "sequence_length",
        "img_size",
        "modality_type",
    ]
    missing = [k for k in required_model_keys if k not in ckpt_args]
    if missing:
        raise RuntimeError(f"Checkpoint args missing required keys: {missing}")
    model_args = SimpleNamespace(**{k: ckpt_args[k] for k in required_model_keys})

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
    from isaaclab.app import AppLauncher  # local import to avoid hard dep on import

    app = AppLauncher(Namespace(**asdict(args))).app

    import isaaclab_tasks  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

    # Create environment (no custom wrapper to preserve dict obs)
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
    # Build policy
    policy = SeqFlowPolicy(state_dim_tr, action_dim, model_args).to(device)
    policy.load_state_dict(model_sd)
    policy.eval()

    # Buffers
    T = int(model_args.sequence_length)
    C, H, W = tuple(model_args.img_size)
    buf = SequenceBuffer(
        T=T, num_envs=args.num_envs, state_dim=state_dim_tr, img_size=(C, H, W)
    ).to(device)

    # Prepare observation adapter using keys from checkpoint if present
    state_keys = ckpt_args.get(
        "state_keys",
        [
            "left_eef_pos",
            "hand_joint_state",
            "right_eef_pos",
            "right_eef_quat",
            "left_eef_quat",
        ],
    )
    image_keys = ckpt_args.get("image_keys", ("robot_pov_cam",))
    image_key = (
        image_keys[0]
        if isinstance(image_keys, (list, tuple)) and len(image_keys) > 0
        else None
    )
    adapter = ObsAdapter(list(state_keys), image_key, (C, H, W))

    # Reset env and prefill buffer with initial obs
    obs_dict, _ = env.reset()
    s0, i0 = adapter.from_obs(obs_dict)
    s0 = s0.to(device)
    i0 = i0.to(device) if i0 is not None else None
    # Fill T times
    for _ in range(T):
        buf.append(s0, i0)

    # Action de-normalization helper
    def denormalize_actions(x: torch.Tensor) -> torch.Tensor:
        if ckpt_args.get("normalize_actions", False) and action_stats is not None:
            a_min = float(action_stats.get("min", 0.0))
            a_max = float(action_stats.get("max", 1.0))
            return x.clamp(0.0, 1.0) * (a_max - a_min) + a_min
        return x

    # Episode/metric trackers
    ep_returns = torch.zeros(args.num_envs, dtype=torch.float32)
    ep_lengths = torch.zeros(args.num_envs, dtype=torch.long)
    completed_returns: List[float] = []
    completed_lengths: List[int] = []
    total_episodes = 0

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
            pred = policy.sample_actions(
                state_seq, image_seq, steps=int(args.val_flow_steps)
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

    wandb.finish()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
