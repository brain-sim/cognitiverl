# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import os
import time
from dataclasses import asdict
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from isaaclab.utils import configclass
from tensordict import TensorDict, from_module
from torch.amp import GradScaler, autocast

import wandb
from scripts.buffers import SimpleReplayBufferOriginal, TorchRLInfoLogger
from scripts.models import AGENT_LOOKUP_BY_ALGORITHM
from scripts.utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    load_args,
    make_isaaclab_env,
    mark_step,
    print_dict,
    save_params,
    seed_everything,
)

# TODO: Batch size (global) and transition batch size should be different.
# The current code only works if they are both the same.


@configclass
class EnvArgs:
    task: str = "Spot-Nav-v0"
    """the id of the environment"""
    env_cfg_entry_point: str = "env_cfg_entry_point"
    """the entry point of the environment configuration"""
    num_envs: int = 4096
    """the number of parallel environments to simulate"""
    seed: int = 1
    """seed of the environment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    video: bool = False
    """record videos during training"""
    video_length: int = 200
    """length of the recorded video (in steps)"""
    video_interval: int = 2000
    """interval between video recordings (in steps)"""
    disable_fabric: bool = False
    """disable fabric and use USD I/O operations"""
    distributed: bool = False
    """run training with multiple GPUs or nodes"""
    headless: bool = False
    """run training in headless mode"""
    enable_cameras: bool = True
    """enable cameras to record sensor inputs."""


@configclass
class ExperimentArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:0"
    """cuda:0 will be enabled by default"""

    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 3e-4
    """the learning rate of the actor"""
    critic_learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 64 * 4
    """the replay memory buffer size"""
    buffer_device: str = "cpu"
    """the device of the replay memory buffer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.1
    """target smoothing coefficient (default: 0.1)"""
    batch_size: int = 16384  # Maximum index limit for a 32 bit integer can count on with image size  128x128x3
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.001
    """the scale of policy noise"""
    learning_starts: int = 10
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    num_updates: int = 4
    """UTD ratio"""

    num_atoms: int = 251
    """the number of atoms for the distributional critic"""
    v_min: float = -150.0
    """the minimum value of the support"""
    v_max: float = 2500.0
    """the maximum value of the support"""
    std_min: float = 0.01
    """the minimum value of the std"""
    std_max: float = 0.5
    """the maximum value of the std"""
    bootstrap: bool = True
    """whether to use bootstrap"""
    use_cdq: bool = True
    """whether to use Clipped Double Q-learning"""
    init_scale: float = 0.01
    """the initial scale of the action"""

    measure_burnin: int = 3

    # Agent config
    obs_type: str = "image"
    """the type of the observations"""
    algorithm: str = "fast_td3"
    """the algorithm to use"""
    img_size: list[int] = [3, 32, 32]
    """the size of the image"""
    obs_normalization: bool = True
    """whether to normalize the observations"""
    reward_normalization: bool = False
    """whether to normalize the rewards"""

    q_chunk: bool = False
    """whether to use q chunk"""
    compile: bool = False
    """use torch.compile"""
    compile_mode: str = "reduce-overhead"
    """the mode of the torch.compile"""
    weight_decay: float = 0.1
    """the weight decay of the optimizer"""
    critic_learning_rate_end: float = 3e-4
    """the learning rate of the critic at the end of the training"""
    actor_learning_rate_end: float = 3e-4
    """the learning rate of the actor at the end of the training"""
    amp: bool = True
    """whether to use amp"""
    amp_dtype: str = "bf16"
    """the dtype of the amp"""
    rollout_length: int = 1
    """the length of the rollout"""
    num_steps: int = 8
    """number of steps to run the environment"""
    action_bounds: float = 1.0
    """the bounds of the action"""
    clip_actions: Optional[float] = None
    """whether to clip the actions"""
    log: bool = False
    """log to wandb"""
    log_interval: int = 100
    """log interval"""
    checkpoint_interval: int = 1000
    """checkpoint interval"""
    load_checkpoint_path: str = ""
    """the path to the checkpoint"""
    resume_from_checkpoint: bool = False
    """whether to resume from checkpoint"""
    save_buffer: bool = False
    """whether to save the buffer"""

    clip_grad_norm: bool = True
    """whether to clip the gradient norm"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    actor_hidden_dims: list[int] = [512, 256, 128]
    """the hidden dimensions of the actor"""
    critic_hidden_dims: list[int] = [1024, 512, 256]
    """the hidden dimensions of the critic"""

    def __post_init__(self):
        if hasattr(self, "num_envs"):
            self.buffer_device = self.device
        self.buffer_size = min(self.buffer_size, self.total_timesteps)


@configclass
class Args(ExperimentArgs, EnvArgs):
    pass


def launch_app(args):
    from argparse import Namespace

    app_launcher = AppLauncher(Namespace(**asdict(args)))
    return app_launcher.app


def get_args():
    return load_args(Args)


try:
    from isaaclab.app import AppLauncher

    args = get_args()
    simulation_app = launch_app(args)
except ImportError:
    raise ImportError("Isaac Lab is not installed. Please install it first.")


def main(args):
    run_name = f"{args.task}__{args.exp_name}"
    amp_enabled = args.amp and "cuda" in args.device and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if "cuda" in args.device and torch.cuda.is_available()
        else "mps"
        if "mps" in args.device and torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    os.environ["WANDB_IGNORE_GLOBS"] = "*.pt,*.mp4"
    run = wandb.init(
        project="fast_td3",
        name=f"{run_name}",
        config=vars(args),
        save_code=True,
    )
    # prepare local checkpoint directory inside the wandb run folder
    run_dir = run.dir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    buffer_device = (
        torch.device(args.buffer_device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # TRY NOT TO MODIFY: seeding
    seed_everything(
        args.seed, use_torch=True, torch_deterministic=args.torch_deterministic
    )
    # env setup
    envs = make_isaaclab_env(
        args.seed,
        args.task,
        args.device,
        args.num_envs,
        args.capture_video,
        args.disable_fabric,
        action_bounds=args.action_bounds,
        clip_actions=args.clip_actions,
        log_dir=run_dir,
        video_length=args.video_length,
        video_interval=args.video_interval,
    )()
    if args.capture_video:
        envs.unwrapped.sim.set_camera_view(eye=[0, 0, 100], target=[0.0, 0.0, 0.0])
    args.img_size = envs.unwrapped.cfg.img_size
    print_dict(args, color="green", attrs=["bold"])
    n_obs = int(np.prod(envs.num_obs))
    n_act = int(np.prod(envs.num_actions))
    action_low, action_high = -1.0, 1.0
    print_dict(f"n_obs: {n_obs}, n_act: {n_act}", color="yellow", attrs=["bold"])
    print_dict(
        f"action_low: {action_low}, action_high: {action_high}",
        color="yellow",
        attrs=["bold"],
    )
    assert isinstance(envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = nn.Identity()

    if args.reward_normalization:
        reward_normalizer = RewardNormalizer(
            gamma=args.gamma,
            device=device,
            g_max=min(abs(args.v_min), abs(args.v_max)),
        )
    else:
        reward_normalizer = nn.Identity()

    # Set horizon for q_chunking
    horizon = 1
    if args.q_chunk:
        horizon = args.num_steps
        print_dict(
            f"Q-chunking enabled with horizon: {horizon}",
            color="cyan",
            attrs=["bold"],
        )

    actor_class, critic_class = AGENT_LOOKUP_BY_ALGORITHM[args.algorithm][args.obs_type]
    actor = actor_class(
        n_obs=n_obs,
        n_act=n_act,
        img_size=args.img_size,
        num_envs=args.num_envs,
        init_scale=args.init_scale,
        std_min=args.std_min,
        std_max=args.std_max,
        device=device,
        dtype=amp_dtype if amp_enabled else torch.float32,
        hidden_dims=args.actor_hidden_dims,
        horizon=horizon,  # Add horizon parameter
    )
    actor_detach = actor_class(
        n_obs=n_obs,
        n_act=n_act,
        img_size=args.img_size,
        num_envs=args.num_envs,
        init_scale=args.init_scale,
        std_min=args.std_min,
        std_max=args.std_max,
        device=device,
        dtype=amp_dtype if amp_enabled else torch.float32,
        hidden_dims=args.actor_hidden_dims,
        horizon=horizon,  # Add horizon parameter
    )
    qnet = critic_class(
        n_obs=n_obs,
        n_act=n_act,
        img_size=args.img_size,
        device=device,
        dtype=amp_dtype if amp_enabled else torch.float32,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dims=args.critic_hidden_dims,
        horizon=horizon,  # Add horizon parameter
    )

    # discard params of net
    qnet_target = critic_class(
        n_obs=n_obs,
        n_act=n_act,
        img_size=args.img_size,
        device=device,
        dtype=amp_dtype if amp_enabled else torch.float32,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dims=args.critic_hidden_dims,
        horizon=horizon,  # Add horizon parameter
    )
    resume_global_step = 0
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.load_checkpoint_path, map_location=device)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        qnet.load_state_dict(checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(checkpoint["qnet_target_state_dict"])
        if "obs_normalizer_state_dict" in checkpoint:
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state_dict"])
        if "critic_obs_normalizer_state_dict" in checkpoint:
            reward_normalizer.load_state_dict(
                checkpoint["critic_obs_normalizer_state_dict"]
            )
        if "reward_normalizer_state_dict" in checkpoint:
            reward_normalizer.load_state_dict(
                checkpoint["reward_normalizer_state_dict"]
            )
        resume_global_step = checkpoint["global_step"]
        print_dict(
            f"Resuming from step: {resume_global_step} which is {resume_global_step * checkpoint['args']['num_envs']} iterations",
            color="magenta",
            attrs=["bold"],
        )

    qnet_target.load_state_dict(qnet.state_dict())

    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    rb = SimpleReplayBufferOriginal(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_obs,
        asymmetric_obs=False,
        q_chunk=args.q_chunk,
        playground_mode=False,
        n_steps=args.num_steps,  # Keep original n_steps
        gamma=args.gamma,
        device=buffer_device,
    )

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def rollout(actions):
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions.float())

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.clone()
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            terminations=infos["terminations"].long(),
            time_outs=infos["time_outs"].long(),
            dones=dones.long(),
            batch_size=(envs.num_envs,),
            device=buffer_device,
        )

        info_logger.update(
            infos,
            transition["observations"].max().item(),
            transition["observations"].min().item(),
            transition["actions"].max().item() * args.action_bounds,
            transition["actions"].min().item() * args.action_bounds,
        )

        # Store transition
        rb.extend(transition)
        return next_obs

    def update_main(data, log_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations, next_observations, actions, rewards, time_outs, dones = (
                data["observations"],
                data["next_observations"],
                data["actions"],
                data["rewards"],
                data["time_outs"],
                data["dones"],
            )

            dones = dones.bool()
            time_outs = time_outs.bool()

            # For q_chunking: target_q(s_t, a_{t:t+h}) = Σ_{t'=t+1}^{t'=h} γ^{t'-t} * r_{t'} + frozen_q(s_{t+h}, a_{t+h:t+2h})
            # The n-step buffer already gives us the discounted sum of rewards
            # Original TD3 logic for non-chunked version
            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )
            if args.bootstrap:
                bootstrap = (time_outs | ~dones).float()
            else:
                bootstrap = (~dones).float()
            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            if next_state_actions.isinf().any():
                raise ValueError("next_state_actions inf")
            discount = args.gamma ** data["effective_n_steps"]
            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )

                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)

                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist = qf1_next_target_projected
                    qf2_next_target_dist = qf2_next_target_projected

            qf1, qf2 = qnet(observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss
        # optimize the model
        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.clip_grad_norm:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)

        scaler.step(q_optimizer)
        scaler.update()

        log_dict["qf_loss"] = qf_loss.detach()
        log_dict["qf1_loss"] = qf1_loss.detach()
        log_dict["qf2_loss"] = qf2_loss.detach()
        log_dict["qf_max"] = qf1.max().detach()
        log_dict["qf_min"] = qf1.min().detach()
        log_dict["batch_rewards"] = rewards.mean().detach()
        log_dict["critic_grad_norm"] = critic_grad_norm.detach()
        return log_dict

    def update_pol(data, log_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            qf1, qf2 = qnet(observations, actor(observations))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=-1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=-1))
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()
        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)

        if args.clip_grad_norm:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)

        scaler.step(actor_optimizer)
        scaler.update()
        log_dict["actor_loss"] = actor_loss.detach()
        log_dict["actor_grad_norm"] = actor_grad_norm.detach()
        return log_dict

    if args.compile:
        compile_mode = args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        update_pol = torch.compile(update_pol, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()

    # Adjust iterations and step size based on q_chunking
    args.num_iterations = args.total_timesteps // (args.num_envs)
    actual_iterations = (
        args.num_iterations - resume_global_step + horizon - 1
    ) // horizon
    pbar = tqdm.tqdm(
        range(resume_global_step, args.num_iterations, horizon),
        initial=resume_global_step,
        total=actual_iterations,
        desc=f"Resuming from step {resume_global_step}",
    )
    start_time = None
    dones = None
    desc = ""
    info_logger = TorchRLInfoLogger(device="cpu", buffer_size=50)
    measure_burnin = None  # Initialize measure_burnin

    for global_step in pbar:
        mark_step()
        out_main = TensorDict()
        if (
            global_step >= args.measure_burnin + resume_global_step
            and measure_burnin is None
        ):
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        with (
            torch.no_grad(),
            autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled),
        ):
            norm_obs = normalize_obs(obs)
            if args.q_chunk:
                actions = policy(obs=norm_obs, dones=dones)
                actions = actions.reshape(actions.shape[0], horizon, -1)
            else:
                actions = policy(obs=norm_obs, dones=dones)
            actions = torch.clamp(actions, action_low, action_high)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = rollout(actions[:, 0] if args.q_chunk else actions)
        log_infos = info_logger.get_averaged_logs()  # TODO
        info_logger.reset()
        wandb.log(
            {
                **log_infos,
            },
            step=global_step * args.num_envs,
        )
        if args.q_chunk:
            for i in range(1, horizon):
                obs = rollout(actions[:, i])
                log_infos = info_logger.get_averaged_logs()  # TODO
                info_logger.reset()
                if i < horizon - 1:
                    wandb.log(
                        {
                            **log_infos,
                        },
                        step=(global_step + i) * args.num_envs,
                    )

        # ALGO LOGIC: training.
        if (
            global_step >= args.measure_burnin + resume_global_step
            and measure_burnin is not None
        ):
            speed = (
                (global_step - measure_burnin)
                * args.num_envs
                / (time.time() - start_time)
            )

        if global_step > args.learning_starts:
            if global_step % args.rollout_length == 0:
                update_start_time = time.time()
                for i in range(args.num_updates * horizon):
                    data = rb.sample(max(1, args.batch_size // args.num_envs))
                    data["observations"] = normalize_obs(data["observations"])
                    data["next_observations"] = normalize_obs(data["next_observations"])
                    data["rewards"] = normalize_reward(data["rewards"])
                    out_main = update_main(data, out_main)
                    if args.num_updates > 1:
                        if i % args.policy_frequency == 1:
                            out_main = update_pol(data, out_main)
                    else:
                        if (
                            global_step // args.rollout_length
                        ) % args.policy_frequency == 0:
                            out_main = update_pol(data, out_main)

                    # update the target networks
                    # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                    soft_update(qnet, qnet_target, args.tau)

                if (
                    args.log_interval > 0
                    and (global_step // args.rollout_length) % args.log_interval == 0
                    and start_time is not None
                ):
                    update_time = time.time() - update_start_time
                    desc = f"update time: {update_time: 4.4f}s"
                    with torch.no_grad():
                        logs = {**log_infos}
                        logs.update({k: v.mean() for k, v in out_main.items()})
                    wandb.log(
                        {
                            "speed": speed,
                            **logs,
                        },
                        step=(global_step + horizon - 1) * args.num_envs
                        if args.q_chunk
                        else global_step * args.num_envs,
                    )

                if (
                    args.checkpoint_interval > 0
                    and (global_step // args.rollout_length) % args.checkpoint_interval
                    == 0
                    and start_time is not None
                ):
                    save_params(
                        global_step=global_step,
                        actor=actor,
                        qnet=qnet,
                        qnet_target=qnet_target,
                        obs_normalizer=obs_normalizer,
                        critic_obs_normalizer=None,
                        args=args,
                        save_path=os.path.join(ckpt_dir, f"ckpt_{global_step}.pt"),
                    )
                actor_scheduler.step()
                q_scheduler.step()

        if (
            global_step >= args.measure_burnin + resume_global_step
            and measure_burnin is not None
        ):
            pbar.set_description(f"{speed: 4.4f} sps, " + desc)

    envs.close()
    save_params(
        global_step=args.total_timesteps // args.num_envs,
        actor=actor,
        qnet=qnet,
        rb=rb if args.save_buffer else None,
        qnet_target=qnet_target,
        args=args,
        obs_normalizer=obs_normalizer,
        critic_obs_normalizer=None,
        save_path=os.path.join(ckpt_dir, "ckpt_final.pt"),
        save_buffer_path=os.path.join(ckpt_dir, "buffer_final.pt")
        if args.save_buffer
        else None,
    )
    if args.log:
        artifact = wandb.Artifact(
            name="fast-td3-final-checkpoint",
            type="model",
            description="Last TD3 with return",
        )
    artifact.add_file(os.path.join(ckpt_dir, "ckpt_final.pt"))
    run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    try:
        if not args.log:
            os.environ["WANDB_MODE"] = "dryrun"
        main(args)
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print("Exception:", e)
    finally:
        simulation_app.close()
