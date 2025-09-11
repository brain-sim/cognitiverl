import argparse
import os
import random
from dataclasses import asdict, fields
from typing import get_type_hints

import gymnasium as gym
import torch.distributed as dist
import torch.nn as nn

try:
    import jax
except ImportError:
    pass

import numpy as np
import torch
import yaml
from termcolor import colored

__all__ = [
    "EmpiricalNormalization",
    "RewardNormalizer",
    "load_args",
    "make_isaaclab_env",
    "mark_step",
    "print_dict",
    "save_params",
    "seed_everything",
    "process_image_batch",  # Add the new function
]


def callable_to_string(func):
    """Convert a callable to a string representation."""
    if hasattr(func, "__name__"):
        return func.__name__
    elif hasattr(func, "__class__"):
        return f"<{func.__class__.__name__} object>"
    else:
        return str(func)


def seed_everything(
    seed,
    use_torch=False,
    use_jax=False,
    torch_deterministic=False,
):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if use_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
    if use_jax:
        key = jax.random.PRNGKey(seed)
        return key


def set_high_precision():
    torch.set_float32_matmul_precision("high")


def _dataclass_to_argparse(parser, dataclass_type, prefix="", defaults=None):
    """
    Add arguments to parser from dataclass fields. Optionally use a prefix for nested dataclasses.
    """
    # Resolve forward references (e.g., from `from __future__ import annotations`)
    try:
        resolved_hints = get_type_hints(dataclass_type)
    except Exception:
        resolved_hints = {}

    for f in fields(dataclass_type):
        arg_name = f"--{prefix}{f.name}"
        arg_type = resolved_hints.get(f.name, f.type)
        # Map string annotations like 'str' to real types
        if isinstance(arg_type, str):
            builtin_map = {"str": str, "int": int, "float": float, "bool": bool}
            arg_type = builtin_map.get(arg_type, str)
        default = getattr(defaults, f.name) if defaults is not None else f.default
        # Handle bools as store_true/store_false
        if arg_type is bool:
            if default is True:
                parser.add_argument(
                    arg_name, action="store_false", dest=f.name, help="(default: True)"
                )
            else:
                parser.add_argument(
                    arg_name, action="store_true", dest=f.name, help="(default: False)"
                )
        else:
            # Fallback to str if the type isn't a callable parser
            parser_type = arg_type if callable(arg_type) else str
            parser.add_argument(
                arg_name,
                type=parser_type,
                default=argparse.SUPPRESS,
                help=f"(default: {default})",
            )


def _update_dict(d, u):
    """Recursively update dict d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_dict(d[k], v)
        else:
            d[k] = v
    return d


def load_args(ArgsClass, yaml_path=None, cli_args=None):
    """
    Loads arguments from dataclass defaults, then YAML, then CLI (in that order).
    Args:
        ArgsClass: The dataclass type to instantiate.
        yaml_path: Path to YAML file (optional, can be None).
        cli_args: List of CLI args (optional, defaults to sys.argv[1:]).
    Returns:
        An instance of ArgsClass with merged arguments.
    """
    # 1. Start with dataclass defaults
    default_args = ArgsClass()
    merged = asdict(default_args)

    # 2. If YAML, update
    if yaml_path is not None and os.path.isfile(yaml_path):
        with open(yaml_path, "r") as f:
            yaml_args = yaml.safe_load(f)
        if yaml_args:
            merged.update(yaml_args)

    # 3. CLI overrides
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file."
    )
    _dataclass_to_argparse(parser, ArgsClass, defaults=default_args)
    args_ns, unknown = parser.parse_known_args(cli_args)
    cli_dict = vars(args_ns)
    config_from_cli = cli_dict.pop("config", None)
    # If --config is given on CLI, reload YAML and update again
    if config_from_cli and (yaml_path is None or config_from_cli != yaml_path):
        if os.path.isfile(config_from_cli):
            with open(config_from_cli, "r") as f:
                yaml_args2 = yaml.safe_load(f)
            if yaml_args2:
                merged.update(yaml_args2)
    merged.update(cli_dict)
    # Return as ArgsClass instance
    return ArgsClass(**merged)


def update_learning_rate_adaptive(
    optimizer, kl_divergence, desired_kl, lr_multiplier, min_lr=1e-6, max_lr=1e-2
):
    current_lr = optimizer.param_groups[0]["lr"]

    if kl_divergence < desired_kl / 100.0:
        new_lr = 1e-9

    if kl_divergence > desired_kl * 2.0:
        new_lr = current_lr / lr_multiplier
    elif kl_divergence < desired_kl / 2.0:
        new_lr = current_lr * lr_multiplier
    else:
        new_lr = current_lr

    # Clamp learning rate to reasonable bounds
    new_lr = np.clip(new_lr, min_lr, max_lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

    return new_lr


def print_dict(
    val,
    nesting: int = -4,
    start: bool = True,
    color: str = "white",
    attrs: list[str] = [],
):
    """Outputs a nested dictionary."""
    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(colored(k, color, attrs=attrs), end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        # deal with functions in print statements
        if callable(val):
            print(colored(callable_to_string(val), color, attrs=attrs))
        else:
            print(colored(val, color, attrs=attrs))


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def make_isaaclab_env(
    seed,
    task,
    device,
    num_envs,
    capture_video,
    disable_fabric,
    log_dir=None,
    max_total_steps=None,
    *args,
    **kwargs,
):
    import cognitiverl.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    from scripts.wrappers import IsaacLabVecEnvWrapper

    def thunk():
        cfg = parse_env_cfg(
            task, device, num_envs=num_envs, use_fabric=not disable_fabric
        )
        cfg.seed = seed
        env = gym.make(
            task,
            cfg=cfg,
            render_mode="rgb_array"
            if (capture_video and log_dir is not None)
            else None,
            max_total_steps=max_total_steps,
        )
        print_dict(
            {"max_episode_steps": env.unwrapped.max_episode_length},
            nesting=4,
            color="green",
            attrs=["bold"],
        )
        print_dict(
            f"capture_video: {capture_video}, log_dir: {log_dir}",
            attrs=["bold"],
            color="magenta",
        )
        if capture_video and log_dir is not None:
            os.makedirs(os.path.join(log_dir, "videos", "play"), exist_ok=True)
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step % kwargs.get("video_interval", 100)
                == 0,
                "video_length": kwargs.get("video_length", 500),
                "fps": 10,
                "disable_logger": True,
            }
            print_dict(video_kwargs, nesting=4, color="magenta", attrs=["bold"])
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env = IsaacLabVecEnvWrapper(
            env
        )  # was earlier set to clip_actions=1.0 causing issues.
        return env

    return thunk


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, center: bool = True, update: bool = True
    ) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}"
            )

        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if torch.jit.is_scripting():
            print("update is being compiled")
            return

        if self.until is not None and self.count >= self.until:
            return

        if dist.is_available() and dist.is_initialized():
            # Calculate global batch size arithmetically
            local_batch_size = x.shape[0]
            world_size = dist.get_world_size()
            global_batch_size = world_size * local_batch_size

            # Calculate the stats
            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            # Sync the stats across all processes
            stats_to_sync = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats_to_sync

            # Calculate the mean and variance of the global batch
            batch_mean_shifted = global_sum_shifted / global_batch_size
            batch_var = (
                global_sum_sq_shifted / global_batch_size - batch_mean_shifted.pow(2)
            )
            batch_mean = batch_mean_shifted + self._mean

        else:
            global_batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))

        # Update variance
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class RewardNormalizer(nn.Module):
    def __init__(
        self,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer(
            "G", torch.zeros(1, device=device)
        )  # running estimate of the discounted return
        self.register_buffer("G_r_max", torch.zeros(1, device=device))  # running-max
        self.G_rms = EmpiricalNormalization(shape=1, device=device)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon

    def _scale_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = self.G_rms.std[0] + self.epsilon
        min_required_denominator = self.G_r_max / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        return rewards / denominator

    def update_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        self.G = self.gamma * (1 - dones) * self.G + rewards
        self.G_rms.update(self.G.view(-1, 1))

        local_max = torch.max(torch.abs(self.G))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

        self.G_r_max = max(self.G_r_max, local_max)

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        return self._scale_reward(rewards)


class PerTaskEmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values for each task."""

    def __init__(
        self,
        num_tasks: int,
        shape: tuple,
        device: torch.device,
        eps: float = 1e-2,
        until: int = None,
    ):
        """
        Initialize PerTaskEmpiricalNormalization module.

        Args:
            num_tasks (int): The total number of tasks.
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If specified, learns until the sum of batch sizes
                                 for a specific task exceeds this value.
        """
        super().__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.num_tasks = num_tasks
        self.shape = shape
        self.eps = eps
        self.until = until
        self.device = device

        # Buffers now have a leading dimension for tasks
        self.register_buffer("_mean", torch.zeros(num_tasks, *shape).to(device))
        self.register_buffer("_var", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer("_std", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer(
            "count", torch.zeros(num_tasks, dtype=torch.long).to(device)
        )

    def forward(
        self, x: torch.Tensor, task_ids: torch.Tensor, center: bool = True
    ) -> torch.Tensor:
        """
        Normalize the input tensor `x` using statistics for the given `task_ids`.

        Args:
            x (torch.Tensor): Input tensor of shape [num_envs, *shape].
            task_ids (torch.Tensor): Tensor of task indices, shape [num_envs].
            center (bool): If True, center the data by subtracting the mean.
        """
        if x.shape[1:] != self.shape:
            raise ValueError(f"Expected input shape (*, {self.shape}), got {x.shape}")
        if x.shape[0] != task_ids.shape[0]:
            raise ValueError("Batch size of x and task_ids must match.")

        # Gather the stats for the tasks in the current batch
        # Reshape task_ids for broadcasting: [num_envs] -> [num_envs, 1, ...]
        view_shape = (task_ids.shape[0],) + (1,) * len(self.shape)
        task_ids_expanded = task_ids.view(view_shape).expand_as(x)

        mean = self._mean.gather(0, task_ids_expanded)
        std = self._std.gather(0, task_ids_expanded)

        if self.training:
            self.update(x, task_ids)

        if center:
            return (x - mean) / (std + self.eps)
        else:
            return x / (std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor, task_ids: torch.Tensor):
        """Update running statistics for the tasks present in the batch."""
        unique_tasks = torch.unique(task_ids)

        for task_id in unique_tasks:
            if self.until is not None and self.count[task_id] >= self.until:
                continue

            # Create a mask to select data for the current task
            mask = task_ids == task_id
            x_task = x[mask]
            batch_size = x_task.shape[0]

            if batch_size == 0:
                continue

            # Update count for this task
            old_count = self.count[task_id].clone()
            new_count = old_count + batch_size

            # Update mean
            task_mean = self._mean[task_id]
            batch_mean = torch.mean(x_task, dim=0)
            delta = batch_mean - task_mean
            self._mean[task_id].copy_(task_mean + (batch_size / new_count) * delta)

            # Update variance using Chan's parallel algorithm
            if old_count > 0:
                batch_var = torch.var(x_task, dim=0, unbiased=False)
                m_a = self._var[task_id] * old_count
                m_b = batch_var * batch_size
                M2 = m_a + m_b + (delta**2) * (old_count * batch_size / new_count)
                self._var[task_id].copy_(M2 / new_count)
            else:
                # For the first batch of this task
                self._var[task_id].copy_(torch.var(x_task, dim=0, unbiased=False))

            self._std[task_id].copy_(torch.sqrt(self._var[task_id]))
            self.count[task_id].copy_(new_count)


class PerTaskRewardNormalizer(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        """
        Per-task reward normalizer, motivation comes from BRC (https://arxiv.org/abs/2505.23150v1)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.device = device

        # Per-task running estimate of the discounted return
        self.register_buffer("G", torch.zeros(num_tasks, device=device))
        # Per-task running-max of the discounted return
        self.register_buffer("G_r_max", torch.zeros(num_tasks, device=device))
        # Use the new per-task normalizer for the statistics of G
        self.G_rms = PerTaskEmpiricalNormalization(
            num_tasks=num_tasks, shape=(1,), device=device
        )

    def _scale_reward(
        self, rewards: torch.Tensor, task_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Scales rewards using per-task statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        # Gather stats for the tasks in the batch
        std_for_batch = self.G_rms._std.gather(0, task_ids.unsqueeze(-1)).squeeze(-1)
        g_r_max_for_batch = self.G_r_max.gather(0, task_ids)

        var_denominator = std_for_batch + self.epsilon
        min_required_denominator = g_r_max_for_batch / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        # Add a small epsilon to the final denominator to prevent division by zero
        # in case g_r_max is also zero.
        return rewards / (denominator + self.epsilon)

    def update_stats(
        self, rewards: torch.Tensor, dones: torch.Tensor, task_ids: torch.Tensor
    ):
        """
        Updates the running discounted return and its statistics for each task.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            dones (torch.Tensor): Done tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        if not (rewards.shape == dones.shape == task_ids.shape):
            raise ValueError("rewards, dones, and task_ids must have the same shape.")

        # === Update G (running discounted return) ===
        # Gather the previous G values for the tasks in the batch
        prev_G = self.G.gather(0, task_ids)
        # Update G for each environment based on its own reward and done signal
        new_G = self.gamma * (1 - dones.float()) * prev_G + rewards
        # Scatter the updated G values back to the main buffer
        self.G.scatter_(0, task_ids, new_G)

        # === Update G_rms (statistics of G) ===
        # The update function handles the per-task logic internally
        self.G_rms.update(new_G.unsqueeze(-1), task_ids)

        # === Update G_r_max (running max of |G|) ===
        prev_G_r_max = self.G_r_max.gather(0, task_ids)
        # Update the max for each environment
        updated_G_r_max = torch.maximum(prev_G_r_max, torch.abs(new_G))
        # Scatter the new maxes back to the main buffer
        self.G_r_max.scatter_(0, task_ids, updated_G_r_max)

    def forward(self, rewards: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Normalizes rewards. During training, it also updates the running statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        return self._scale_reward(rewards, task_ids)


def cpu_state(sd):
    # detach & move to host without locking the compute stream
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def save_params(
    global_step,
    actor,
    qnet,
    qnet_target,
    args,
    save_path,
    rb=None,
    obs_normalizer=None,
    critic_obs_normalizer=None,
    save_buffer_path=None,
):
    """Save model parameters and training configuration to disk."""

    def get_ddp_state_dict(model):
        """Get state dict from model, handling DDP wrapper if present."""
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(get_ddp_state_dict(actor)),
        "qnet_state_dict": cpu_state(get_ddp_state_dict(qnet)),
        "qnet_target_state_dict": cpu_state(get_ddp_state_dict(qnet_target)),
        "args": vars(args),  # Save all arguments
        "global_step": global_step,
    }
    if obs_normalizer is not None:
        save_dict["obs_normalizer_state"] = (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        )
    if critic_obs_normalizer is not None:
        save_dict["critic_obs_normalizer_state"] = (
            cpu_state(critic_obs_normalizer.state_dict())
            if hasattr(critic_obs_normalizer, "state_dict")
            else None
        )
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    if save_buffer_path is not None and rb is not None:
        torch.save(rb, save_buffer_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")


def get_ddp_state_dict(model):
    """Get state dict from model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def load_ddp_state_dict(model, state_dict):
    """Load state dict into model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


@torch.no_grad()
def mark_step():
    # call this once per iteration *before* any compiled function
    torch.compiler.cudagraph_mark_step_begin()


def adjust_noise_scales(state_dict, agent, num_eval_envs):
    """Adjust noise_scales parameter to match the current number of environments"""
    if "noise_scales" in state_dict:
        checkpoint_noise_scales = state_dict["noise_scales"]
        current_noise_scales = agent.noise_scales

        if checkpoint_noise_scales.shape != current_noise_scales.shape:
            print(
                f"Adjusting noise_scales from {checkpoint_noise_scales.shape} to {current_noise_scales.shape}"
            )
            # If num_eval_envs > checkpoint envs, repeat the noise scales
            if num_eval_envs > checkpoint_noise_scales.shape[0]:
                # Repeat the noise scales to match num_eval_envs
                repeats_needed = (
                    num_eval_envs + checkpoint_noise_scales.shape[0] - 1
                ) // checkpoint_noise_scales.shape[0]
                repeated_scales = checkpoint_noise_scales.repeat(repeats_needed, 1)
                state_dict["noise_scales"] = repeated_scales[:num_eval_envs]
            else:
                # Take the first num_eval_envs from the checkpoint
                state_dict["noise_scales"] = checkpoint_noise_scales[:num_eval_envs]
    return state_dict
