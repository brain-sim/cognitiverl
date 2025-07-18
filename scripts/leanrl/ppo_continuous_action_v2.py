# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
import wandb
from termcolor import colored
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaaclab.utils import configclass
from models import CNNPPOAgent, MLPPPOAgent
from utils import load_args, print_dict, seed_everything, update_learning_rate_adaptive


@configclass
class EnvArgs:
    task: str = "Spot-Nav-v0"  # """the id of the environment"""
    env_cfg_entry_point: str = (
        "env_cfg_entry_point"  # """the entry point of the environment configuration"""
    )
    num_envs: int = 4096  # """the number of parallel environments to simulate"""
    seed: int = 1  # """seed of the environment"""
    capture_video: bool = True  # """whether to capture videos of the agent performances (check out `videos` folder)"""
    video: bool = False  # """record videos during training"""
    video_length: int = 200  # """length of the recorded video (in steps)"""
    video_interval: int = 2000  # """interval between video recordings (in steps)"""
    disable_fabric: bool = False  # """disable fabric and use USD I/O operations"""
    distributed: bool = False  # """run training with multiple GPUs or nodes"""
    headless: bool = False  # """run training in headless mode"""
    enable_cameras: bool = False  # """enable cameras to record sensor inputs."""

    # renderer: str = "PathTracing"
    # """Renderer to use."""
    # samples_per_pixel_per_frame: int = 1
    # """Number of samples per pixel per frame."""


@configclass
class ExperimentArgs:
    exp_name: str = os.path.basename(__file__)[
        : -len(".py")
    ]  # """the name of this experiment"""
    torch_deterministic: bool = (
        False  # """if toggled, `torch.backends.cudnn.deterministic=False`"""
    )
    device: str = "cuda:0"  # """device to use for training"""
    dtype: str = "bfloat16"  # "float32", "float64", "bfloat16"
    use_amp: bool = False  # Mixed precision training
    wandb_project: str = (
        "ppo_continuous_action"  # """wandb project to use for training"""
    )

    # Algorithm specific arguments

    total_timesteps: int = 10_000_000  # """total timesteps of the experiments"""
    learning_rate: float = 0.0003  # 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 64  # 64
    anneal_lr: bool = (
        False  # """Toggle learning rate annealing for policy and value networks"""
    )
    gamma: float = 0.99  # """the discount factor gamma"""
    gae_lambda: float = 0.95  # """the lambda for the general advantage estimation"""
    num_minibatches: int = 4  # 4 """the number of mini-batches"""
    update_epochs: int = 10  # 10 """the K epochs to update the policy"""
    norm_adv: bool = False  # """Toggles advantages normalization"""
    clip_coef: float = 0.2  # """the surrogate clipping coefficient"""
    clip_vloss: bool = True  # """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0  # 0.0 """coefficient of the entropy"""
    vf_coef: float = 0.5  # 0.5 """coefficient of the value function"""
    max_grad_norm: float = 1.0  # """the maximum norm for the gradient clipping"""
    init_at_random_ep_len: bool = (
        False  # """randomize initial episode lengths (for exploration)"""
    )

    # EMA parameters
    use_ema: bool = True  # """Enable Exponential Moving Average for model weights"""
    ema_decay: float = 0.95  # """EMA decay rate for model weights"""
    ema_start_step: int = (
        10_000  # """Start applying EMA after this many global steps"""
    )
    use_ema_for_eval: bool = (
        True  # """Use EMA weights for evaluation and checkpointing"""
    )

    # to be filled in runtime
    batch_size: int = 0  # """the batch size (computed in runtime)"""
    minibatch_size: int = 0  # """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0  # """the number of iterations (computed in runtime)"""

    measure_burnin: int = 3  # """the number of iterations to measure burnin"""

    # Agent config
    agent_type: str = "CNNPPOAgent"  # """the type of agent to use"""

    checkpoint_interval: int = (
        total_timesteps  # """environment steps between saving checkpoints."""
    )
    num_eval_envs: int = 3  # """number of environments to run for evaluation/play."""
    num_eval_env_steps: int = 200  # """number of steps to run for evaluation/play."""
    log_interval: int = 10  # """number of iterations between logging."""
    log: bool = False  # """whether to log the training process."""
    log_video: bool = False  # """whether to log the video."""

    # Adaptive learning rate parameters
    adaptive_lr: bool = True  # """Use adaptive learning rate based on KL divergence"""
    target_kl: float = 0.01  # """the target KL divergence threshold"""
    lr_multiplier: float = 1.5  # """Factor to multiply/divide learning rate by"""

    img_size: List[int] = [3, 32, 32]  # """the size of the image"""
    resume_from_checkpoint: str = ""  # """the path to the checkpoint to resume from"""

    use_amp: bool = False  # """Use mixed precision training"""


@configclass
class Args(ExperimentArgs, EnvArgs):
    pass


class PPOTrainer:
    """Clean PPO Trainer without distributed logic."""

    def __init__(
        self,
        agent,
        envs,
        args,
        optimizer: Optional[torch.optim.Optimizer] = None,
        show_progress: bool = True,  # Add parameter to control progress bars
    ):
        self.agent = agent
        self.envs = envs
        self.args = args
        self.device = agent.device
        self.dtype = self._get_dtype(args.dtype)
        self.show_progress = show_progress  # Store progress bar setting

        if self.show_progress:
            self.step_bar = tqdm.tqdm(
                total=args.num_steps,
                desc="Collecting rollouts",
                position=1,
                leave=False,
            )

        # Setup optimizer
        self.optimizer = optimizer or optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=1e-5
        )

        # Setup mixed precision
        self.use_amp = args.use_amp
        self.grad_scaler = GradScaler() if self.use_amp else None

        # Setup EMA
        if args.use_ema:
            self.ema_agent = self.agent.create_ema_agent(args.ema_decay)
        else:
            self.ema_agent = None

        # Storage
        self._init_storage()

        # Tracking
        self.global_step = 0
        self.best_return = -float("inf")
        self.best_ckpt = None

        # Logging buffers
        self.reward_buffer = {}
        self.metric_buffer = {}
        self.curriculum_buffer = {}
        self.termination_buffer = {}

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        return dtype_map.get(dtype_str, torch.float32)

    def _init_storage(self):
        """Initialize rollout storage."""
        args = self.args
        self.obs = torch.zeros(
            (args.num_steps,) + self.envs.observation_space.shape, dtype=self.dtype
        ).to(self.device)
        self.actions = torch.zeros(
            (args.num_steps,) + self.envs.action_space.shape, dtype=self.dtype
        ).to(self.device)
        self.logprobs = torch.zeros(
            (args.num_steps, args.num_envs), dtype=self.dtype
        ).to(self.device)
        self.rewards = torch.zeros(
            (args.num_steps, args.num_envs), dtype=self.dtype
        ).to(self.device)
        self.dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.uint8).to(
            self.device
        )
        self.values = torch.zeros((args.num_steps, args.num_envs), dtype=self.dtype).to(
            self.device
        )
        self.mus = torch.zeros_like(self.actions)
        self.sigmas = torch.zeros_like(self.actions)

        print("class dtypes")
        print(
            f"obs: {self.obs.dtype} | actions: {self.actions.dtype} | logprobs: {self.logprobs.dtype} | rewards: {self.rewards.dtype} | dones: {self.dones.dtype} | values: {self.values.dtype} | mus: {self.mus.dtype} | sigmas: {self.sigmas.dtype}"
        )

    def get_inference_agent(self):
        """Get agent for inference (EMA or regular)."""
        if (
            self.args.use_ema
            and self.ema_agent is not None
            and self.global_step >= self.args.ema_start_step
        ):
            return self.ema_agent
        return self.agent

    @torch.inference_mode()
    def collect_rollouts(
        self, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, float, List]:
        """Collect rollouts from environment."""
        args = self.args
        inference_agent = self.get_inference_agent()
        video_frames = []

        if args.log_video:
            indices = torch.randperm(args.num_envs)[: min(9, args.num_envs)].to(
                self.device
            )

        # Create progress bar only if show_progress is True
        if self.show_progress:
            self.step_bar.reset()

        start_time = time.time()

        for step in range(args.num_steps):
            self.global_step += args.num_envs
            self.obs[step] = next_obs.to(dtype=self.dtype)

            # Get action with optional mixed precision
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=torch.float16):
                    action, logprob, _, value, mu, sigma = (
                        inference_agent.get_action_and_value(self.obs[step])
                    )
            else:
                action, logprob, _, value, mu, sigma = (
                    inference_agent.get_action_and_value(self.obs[step])
                )

            self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob
            self.mus[step] = mu
            self.sigmas[step] = sigma

            # Environment step
            try:
                next_obs, reward, next_done, infos = self.envs.step(action)
            except Exception:
                action = action.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                next_obs, reward, next_done, infos = self.envs.step(action)

            print(
                f"next_obs : {next_obs.dtype} | reward: {reward.dtype} | action: {action.dtype} | next_done: {next_done.dtype}"
            )

            self.rewards[step] = reward.view(-1).to(dtype=self.dtype)
            self.dones[step] = next_done.to(dtype=torch.uint8)

            # Process logging info
            self._process_infos(infos)

            # Collect video frames
            if args.log_video:
                frame = next_obs[indices, : 3 * 32 * 32].reshape(-1, 3, 32, 32)
                frame = (
                    torchvision.utils.make_grid(frame, nrow=3, scale_each=True) * 255.0
                )
                video_frames.append(frame)

            step_speed = (step + 1) * args.num_envs / (time.time() - start_time)

            # Update progress bar
            if self.show_progress:
                self.step_bar.update(1)
                self.step_bar.set_description(
                    f"global_step: {self.global_step:,}, speed: {step_speed:.1f} (sps)"
                )

        return next_obs.to(dtype=self.dtype), step_speed, video_frames

    def close_step_bar(self):
        if self.show_progress:
            self.step_bar.close()

    def _process_infos(self, infos):
        """Process step information for logging."""
        if "log" not in infos:
            return

        log_data = infos["log"]

        # Process different types of logged information
        for key, value in log_data.items():
            if key.startswith("Episode_Reward/"):
                name = key.replace("Episode_Reward/", "")
                self.reward_buffer.setdefault(name, []).append(value)
            elif key.startswith("Metrics/"):
                name = key.replace("Metrics/", "")
                self.metric_buffer.setdefault(name, []).append(value)
            elif key.startswith("Curriculum/"):
                name = key.replace("Curriculum/", "")
                self.curriculum_buffer.setdefault(name, []).append(value)
            elif key.startswith("Episode_Termination/"):
                name = key.replace("Episode_Termination/", "")
                self.termination_buffer.setdefault(name, []).append(value)

    @torch.inference_mode()
    def compute_advantages(
        self, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages."""
        args = self.args
        inference_agent = self.get_inference_agent()

        # Bootstrap value
        if self.use_amp:
            with autocast(device_type=self.device.type, dtype=torch.float16):
                next_value = inference_agent.get_value(next_obs).reshape(1, -1)
        else:
            next_value = inference_agent.get_value(next_obs).reshape(1, -1)

        returns = torch.zeros_like(self.rewards)
        advantage = 0

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextvalues = next_value
            else:
                nextvalues = self.values[t + 1]
            nextnonterminal = 1.0 - self.dones[t].to(dtype=self.dtype)
            delta = (
                self.rewards[t]
                + args.gamma * nextvalues * nextnonterminal
                - self.values[t]
            )
            advantage = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * advantage
            )
            returns[t] = advantage + self.values[t]

        advantages = returns - self.values
        return returns, advantages

    def update_policy(self, returns: torch.Tensor, advantages: torch.Tensor) -> dict:
        """Update policy using PPO."""
        args = self.args

        # Flatten batch
        b_obs = self.obs.reshape((-1,) + self.envs.observation_space.shape[1:])
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.action_space.shape[1:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_mus = self.mus.reshape((-1,) + self.envs.action_space.shape[1:])
        b_sigmas = self.sigmas.reshape((-1,) + self.envs.action_space.shape[1:])

        # Update epochs
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Forward pass with optional mixed precision
                if self.use_amp:
                    with autocast(device_type=self.device.type, dtype=torch.float16):
                        _, newlogprob, entropy, newvalue, newmu, newsigma = (
                            self.agent.get_action_and_value(
                                b_obs[mb_inds], b_actions[mb_inds]
                            )
                        )
                else:
                    _, newlogprob, entropy, newvalue, newmu, newsigma = (
                        self.agent.get_action_and_value(
                            b_obs[mb_inds], b_actions[mb_inds]
                        )
                    )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # KL divergence for monitoring
                with torch.inference_mode():
                    kl_mean = torch.mean(
                        torch.sum(
                            torch.log(newsigma / (b_sigmas[mb_inds] + 1e-8) + 1e-8)
                            + (
                                torch.square(b_sigmas[mb_inds])
                                + torch.square(b_mus[mb_inds] - newmu)
                            )
                            / (2 * torch.square(newsigma) + 1e-8)
                            - 0.5,
                            dim=-1,
                        )
                    )
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Backward pass
                self.optimizer.zero_grad()

                if self.use_amp and self.grad_scaler is not None:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.agent.parameters(), args.max_grad_norm
                    )
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.agent.parameters(), args.max_grad_norm
                    )
                    self.optimizer.step()

                # Adaptive learning rate
                if args.target_kl is not None and args.adaptive_lr:
                    update_learning_rate_adaptive(
                        self.optimizer,
                        kl_mean.item(),
                        args.target_kl,
                        args.lr_multiplier,
                    )

        # Update EMA
        if (
            args.use_ema
            and self.ema_agent is not None
            and self.global_step >= args.ema_start_step
        ):
            self.agent.update_ema(args.ema_decay)

        # Compute explained variance
        y_pred, y_true = b_values.cpu().float().numpy(), b_returns.cpu().float().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "kl_mean": kl_mean.item(),
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            "explained_var": explained_var,
        }

    def log_metrics(self, step_speed: float, update_metrics: dict) -> dict:
        """Prepare metrics for logging."""
        metrics = {
            "speed": step_speed,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "global_step": self.global_step,
            **update_metrics,
        }

        # Add buffered metrics
        for name, values in self.reward_buffer.items():
            if values:
                metrics[f"rewards/{name}"] = torch.tensor(values).float().mean().item()

        for name, values in self.metric_buffer.items():
            if values:
                metrics[f"metrics/{name}"] = torch.tensor(values).float().mean().item()

        for name, values in self.curriculum_buffer.items():
            if values:
                metrics[f"curriculum/{name}"] = (
                    torch.tensor(values).float().mean().item()
                )

        for name, values in self.termination_buffer.items():
            if values:
                metrics[f"terminations/{name}"] = (
                    torch.tensor(values).float().mean().item()
                )

        # Clear buffers
        self.reward_buffer.clear()
        self.metric_buffer.clear()
        self.curriculum_buffer.clear()
        self.termination_buffer.clear()

        return metrics

    def save_checkpoint(self, ckpt_dir: str, iteration: int) -> str:
        """Save checkpoint."""
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_{self.global_step}.pt")
        self.agent.save_checkpoint(
            ckpt_path,
            optimizer=self.optimizer,
            step=self.global_step,
            iteration=iteration,
            args=vars(self.args),
        )
        return ckpt_path


class DistributedTrainer:
    """Distributed wrapper around PPO Trainer."""

    def __init__(self, ppo_trainer: PPOTrainer, rank: int = 0, world_size: int = 1):
        self.ppo_trainer = ppo_trainer
        self.rank = rank
        self.world_size = world_size

        # Only show progress bars on rank 0 for distributed training
        if world_size > 1:
            self.ppo_trainer.show_progress = rank == 0

        # Wrap agent with DDP if distributed
        if world_size > 1:
            self.ppo_trainer.agent = DDP(
                ppo_trainer.agent, device_ids=[rank], find_unused_parameters=False
            )

    @property
    def global_step(self):
        """Access global step from PPO trainer."""
        return self.ppo_trainer.global_step

    @property
    def args(self):
        """Access args from PPO trainer."""
        return self.ppo_trainer.args

    def collect_rollouts(
        self, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, float, List]:
        """Collect rollouts (delegates to PPO trainer)."""
        return self.ppo_trainer.collect_rollouts(next_obs)

    def compute_advantages(
        self, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages (delegates to PPO trainer)."""
        return self.ppo_trainer.compute_advantages(next_obs)

    def update_policy(self, returns: torch.Tensor, advantages: torch.Tensor) -> dict:
        """Update policy with distributed support."""
        # For DDP, we need to access the module
        if self.world_size > 1:
            # Temporarily store original agent
            original_agent = self.ppo_trainer.agent
            # Access the underlying module for PPO trainer
            self.ppo_trainer.agent = original_agent.module

        # Run PPO update
        update_metrics = self.ppo_trainer.update_policy(returns, advantages)

        # Restore DDP wrapper
        if self.world_size > 1:
            self.ppo_trainer.agent = original_agent

        return update_metrics

    def log_metrics(self, step_speed: float, update_metrics: dict) -> dict:
        """Log metrics (only on rank 0)."""
        if self.rank == 0:
            return self.ppo_trainer.log_metrics(step_speed, update_metrics)
        return {}

    def save_checkpoint(self, ckpt_dir: str, iteration: int) -> Optional[str]:
        """Save checkpoint (only on rank 0)."""
        if self.rank == 0:
            # Access base agent for checkpointing
            base_agent = self.ppo_trainer.agent
            if self.world_size > 1:
                base_agent = self.ppo_trainer.agent.module

            # Temporarily set base agent for saving
            original_agent = self.ppo_trainer.agent
            self.ppo_trainer.agent = base_agent

            ckpt_path = self.ppo_trainer.save_checkpoint(ckpt_dir, iteration)

            # Restore DDP wrapper
            self.ppo_trainer.agent = original_agent

            return ckpt_path
        return None

    def train_step(
        self, next_obs: torch.Tensor, iteration: int
    ) -> Tuple[torch.Tensor, dict, List]:
        """Execute one complete training step."""
        # Anneal learning rate
        if self.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
            lrnow = frac * self.args.learning_rate
            self.ppo_trainer.optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollouts
        next_obs, step_speed, video_frames = self.collect_rollouts(next_obs)

        # Compute advantages
        returns, advantages = self.compute_advantages(next_obs)

        # Update policy
        update_metrics = self.update_policy(returns, advantages)

        # Prepare metrics for logging
        metrics = self.log_metrics(step_speed, update_metrics)

        return next_obs, metrics, video_frames


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def make_isaaclab_env(
    task: str,
    device: str,
    num_envs: int,
    capture_video: bool,
    disable_fabric: bool,
    dtype_str: str = "float32",
    log_dir: str | None = None,
    max_total_steps: int | None = None,
    **kwargs,
):
    """Create IsaacLab environment with precision wrapper."""
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.torchrl import IsaacLabVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import cognitiverl.tasks  # noqa: F401

    def thunk():
        cfg = parse_env_cfg(
            task, device, num_envs=num_envs, use_fabric=not disable_fabric
        )
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
        # Add precision wrapper
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        dtype = dtype_map.get(dtype_str, torch.float32)
        env = IsaacLabVecEnvWrapper(
            env,
            dtype=dtype,
        )  # was earlier set to clip_actions=1.0 causing issues.

        return env

    return thunk


def launch_app(args):
    from argparse import Namespace

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(Namespace(**asdict(args)))
    return app_launcher.app


def main_worker(rank: int, world_size: int, args):
    """Main training worker for distributed training."""
    if world_size > 1:
        setup_distributed(rank, world_size)

    # Update device for distributed training
    if world_size > 1:
        args.device = f"cuda:{rank}"

    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    run_name = f"{args.task}__{args.exp_name}__{args.seed}"
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Initialize wandb (only on rank 0)
    if rank == 0:
        if not args.log:
            os.environ["WANDB_MODE"] = "dryrun"

        run = wandb.init(
            project=args.wandb_project,
            name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
            config=vars(args),
            save_code=True,
        )
        ckpt_dir = os.path.join(run.dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        ckpt_dir = None

    # Environment setup
    envs = make_isaaclab_env(
        args.task,
        args.device,
        args.num_envs,
        args.capture_video,
        args.disable_fabric,
        dtype_str=args.dtype,
    )()

    # Seeding
    seed_everything(envs, args.seed, use_torch=True, torch_deterministic=True)

    n_obs = int(np.prod(envs.observation_space.shape[1:]))
    n_act = int(np.prod(envs.action_space.shape[1:]))

    # Create agent
    if args.agent_type == "CNNPPOAgent":
        args.img_size = envs.unwrapped.cfg.img_size
        agent = CNNPPOAgent(
            n_obs=n_obs,
            n_act=n_act,
            img_size=envs.unwrapped.cfg.img_size,
            device=device,
            dtype=dtype,
        )
        if args.resume_from_checkpoint:
            agent.load_checkpoint(args.resume_from_checkpoint)
    else:
        agent = MLPPPOAgent(n_obs=n_obs, n_act=n_act, device=device, dtype=dtype)
        if args.resume_from_checkpoint:
            agent.load_checkpoint(args.resume_from_checkpoint)

    if rank == 0:
        print_dict(colored(args.__dict__, "green", attrs=["bold"]), nesting=4)

    # Create PPO trainer with progress bar control
    ppo_trainer = PPOTrainer(
        agent,
        envs,
        args,
        show_progress=(rank == 0),  # Only show progress on rank 0
    )

    # Wrap with distributed trainer
    trainer = DistributedTrainer(ppo_trainer, rank=rank, world_size=world_size)

    # Training loop
    next_obs, _ = envs.reset()
    next_obs = next_obs.to(dtype=dtype)

    if args.init_at_random_ep_len:
        envs.unwrapped.episode_length_buf = torch.randint_like(
            envs.unwrapped.episode_length_buf,
            high=int(envs.unwrapped.max_episode_length),
        )

    # Progress bars (only on rank 0)
    if rank == 0:
        iteration_pbar = tqdm.tqdm(total=args.num_iterations, desc="Iterations")

    global_step_burnin = None

    for iteration in range(1, args.num_iterations + 1):
        if iteration == args.measure_burnin:
            global_step_burnin = trainer.global_step

        # Training step - no need to reset step_pbar here anymore
        next_obs, metrics, video_frames = trainer.train_step(next_obs, iteration)

        # Logging (only on rank 0)
        if rank == 0:
            iteration_pbar.update(1)
            iteration_pbar.set_description(
                f"ret: {metrics.get('metrics/episode_reward', 0):.2f} | "
                f"max_ret: {metrics.get('metrics/max_episode_return', 0):.2f} | "
                f"step_rew: {metrics.get('metrics/avg_step_reward', 0):.2f} | "
                f"max_step_rew: {metrics.get('metrics/max_step_reward', 0):.2f}"
            )
            # Log to wandb
            if global_step_burnin is not None:
                if iteration % args.checkpoint_interval == 0:
                    ckpt_path = trainer.save_checkpoint(ckpt_dir, iteration)
                    if ckpt_path:
                        print(
                            colored(
                                f"Checkpoint saved: {ckpt_path}",
                                "magenta",
                                attrs=["bold"],
                            )
                        )
                if iteration % args.log_interval == 0:
                    wandb.log(metrics, step=trainer.global_step)

                    # Log video
                    if args.log_video and video_frames:
                        video_tensor = torch.stack(video_frames)
                        wandb.log(
                            {
                                "obs_grid_video": wandb.Video(
                                    video_tensor.detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.uint8),
                                    fps=25,
                                    format="mp4",
                                )
                            },
                            step=trainer.global_step,
                        )

    # Cleanup
    if rank == 0:
        iteration_pbar.close()
        ppo_trainer.close_step_bar()
        wandb.finish()

    envs.close()

    if world_size > 1:
        dist.destroy_process_group()


def main():
    """Main function."""
    args = load_args(Args)
    if not args.log:
        os.environ["WANDB_MODE"] = "dryrun"

    try:
        from isaaclab.app import AppLauncher

        simulation_app = launch_app(args)
    except ImportError:
        raise ImportError("Isaac Lab is not installed. Please install it first.")

    try:
        # Check for distributed training
        world_size = torch.cuda.device_count() if args.distributed else 1

        if world_size > 1:
            import torch.multiprocessing as mp

            mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
        else:
            main_worker(0, 1, args)

    except Exception as e:
        print(f"Training error: {e}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()


# Usage examples:
# Single GPU: python ppo_continuous_action_clean.py --task Spot-Nav-v0 --dtype bfloat16
# Multi-GPU: python ppo_continuous_action_clean.py --task Spot-Nav-v0 --distributed --dtype bfloat16
