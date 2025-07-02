# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import time
from dataclasses import asdict
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
import wandb
from termcolor import colored

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaaclab.utils import configclass
from models import CNNPPOAgent, MLPPPOAgent
from utils import load_args, print_dict, seed_everything, update_learning_rate_adaptive


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
    capture_video: bool = True
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
    enable_cameras: bool = False
    """enable cameras to record sensor inputs."""

    # renderer: str = "PathTracing"
    # """Renderer to use."""
    # samples_per_pixel_per_frame: int = 1
    # """Number of samples per pixel per frame."""


@configclass
class ExperimentArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:0"
    """device to use for training"""
    wandb_project: str = "ppo_continuous_action"
    """wandb project to use for training"""

    # Algorithm specific arguments

    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0003  # 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 64  # 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4  # 4
    """the number of mini-batches"""
    update_epochs: int = 10  # 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0  # 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5  # 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    init_at_random_ep_len: bool = False
    """randomize initial episode lengths (for exploration)"""

    # EMA parameters
    use_ema: bool = False
    """Enable Exponential Moving Average for model weights"""
    ema_decay: float = 0.95
    """EMA decay rate for model weights"""
    ema_start_step: int = 10_000
    """Start applying EMA after this many global steps"""
    use_ema_for_eval: bool = True
    """Use EMA weights for evaluation and checkpointing"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 3

    # Agent config
    agent_type: str = "CNNPPOAgent"

    checkpoint_interval: int = 10
    """environment steps between saving checkpoints."""
    num_eval_envs: int = 3
    """number of environments to run for evaluation/play."""
    num_eval_env_steps: int = 200
    """number of steps to run for evaluation/play."""
    log_interval: int = 10
    """number of iterations between logging."""
    log: bool = False
    """whether to log the training process."""
    log_video: bool = False
    """whether to log the video."""

    # Adaptive learning rate parameters
    adaptive_lr: bool = True
    """Use adaptive learning rate based on KL divergence"""
    target_kl: float = 0.01
    """the target KL divergence threshold"""
    lr_multiplier: float = 1.5
    """Factor to multiply/divide learning rate by"""

    img_size: List[int] = [3, 32, 32]
    """the size of the image"""
    resume_from_checkpoint: str = ""
    """the path to the checkpoint to resume from"""


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
    task,
    device,
    num_envs,
    capture_video,
    disable_fabric,
    log_dir=None,
    video_length=200,
    max_total_steps=None,
    *args,
    **kwargs,
):
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.torchrl import (
        IsaacLabVecEnvWrapper,
    )
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
        env = IsaacLabVecEnvWrapper(
            env
        )  # was earlier set to clip_actions=1.0 causing issues.

        if capture_video and log_dir is not None:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": video_length,
                "disable_logger": True,
            }
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        return env

    return thunk


# Add EMA helper functions after imports
import copy


def create_ema_agent(agent):
    """Create EMA copy of the agent"""
    ema_agent = copy.deepcopy(agent)
    ema_agent.eval()  # Set to eval mode
    # Disable gradients for EMA agent
    for param in ema_agent.parameters():
        param.requires_grad = False
    return ema_agent


def update_ema_weights(ema_agent, agent, decay):
    """Update EMA weights"""
    with torch.no_grad():
        for ema_param, param in zip(ema_agent.parameters(), agent.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def main(args):
    print_dict(colored(args, "green", attrs=["bold"]), nesting=4)
    run_name = f"{args.task}__{args.exp_name}__{args.seed}"

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # initialize wandb run
    run = wandb.init(
        project=args.wandb_project if args.wandb_project else "ppo_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # prepare local checkpoint directory inside the wandb run folder
    run_dir = run.dir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # tracking best model
    best_return = -float("inf")
    best_step = 0
    best_ckpt = None

    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )

    # env setup
    envs = make_isaaclab_env(
        args.task,
        args.device,
        args.num_envs,
        args.capture_video,
        args.disable_fabric,
        max_total_steps=args.total_timesteps,
    )()
    # TRY NOT TO MODIFY: seeding
    seed_everything(
        envs,
        args.seed,
        use_torch=True,
        torch_deterministic=True,
    )
    n_obs = int(np.prod(envs.observation_space.shape[1:]))
    n_act = int(np.prod(envs.action_space.shape[1:]))
    print("n_obs:", n_obs)
    print("n_act:", n_act)
    assert isinstance(envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    if args.agent_type == "CNNPPOAgent":
        print("img_size:", envs.unwrapped.cfg.img_size)
        agent = CNNPPOAgent(n_obs, n_act, img_size=envs.unwrapped.cfg.img_size)
        if args.resume_from_checkpoint:
            agent.load_from_checkpoint(args.resume_from_checkpoint)
            print(
                colored(
                    f"Loaded checkpoint from {args.resume_from_checkpoint}",
                    "green",
                    attrs=["bold"],
                )
            )
        agent.to(device)
    else:
        agent = MLPPPOAgent(n_obs, n_act)
        if args.resume_from_checkpoint:
            agent.load_from_checkpoint(args.resume_from_checkpoint)
            print(
                colored(
                    f"Loaded checkpoint from {args.resume_from_checkpoint}",
                    "green",
                    attrs=["bold"],
                )
            )
        agent.to(device)

    # Create EMA agent if enabled
    ema_agent = None
    if args.use_ema:
        ema_agent = create_ema_agent(agent)
        print(
            colored(
                f"EMA agent created with decay={args.ema_decay}",
                "green",
                attrs=["bold"],
            )
        )

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps,) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps,) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    mus = torch.zeros_like(actions)
    sigmas = torch.zeros_like(actions)
    # TRY NOT TO MODIFY: start the game
    global_step = 0

    next_obs, _ = envs.reset()

    # randomize initial episode lengths (for exploration)
    if args.init_at_random_ep_len:
        envs.unwrapped.episode_length_buf = torch.randint_like(
            envs.unwrapped.episode_length_buf,
            high=int(envs.unwrapped.max_episode_length),
        )

    avg_returns, ep_ret, max_ep_ret, max_step_reward = 0.0, 0.0, 0.0, 0.0
    # Create two static progress bars
    iteration_pbar = tqdm.tqdm(
        total=args.num_iterations, desc="Iterations", position=0, leave=True
    )
    step_pbar = tqdm.tqdm(total=args.num_steps, desc="Steps", position=1, leave=True)

    global_step_burnin = None
    start_time = None
    desc = ""

    if args.log_video:
        video_frames = []
        # Randomly choose minimum of 9 environments or all environments
        indices = torch.randperm(args.num_envs)[: min(9, args.num_envs)].to(device)

    # Add these buffers before the training loop starts
    reward_info_buffer = {}
    metric_info_buffer = {}
    curriculum_info_buffer = {}
    termination_info_buffer = {}

    for iteration in range(1, args.num_iterations + 1):
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Reset step progress bar for each iteration
        step_pbar.reset()
        inference_agent = (
            ema_agent
            if (
                args.use_ema
                and ema_agent is not None
                and global_step >= args.ema_start_step
            )
            else agent
        )

        with torch.inference_mode():
            start_time = time.time()
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs

                # ALGO LOGIC: action logic
                action, logprob, _, value, mu, sigma = (
                    inference_agent.get_action_and_value(next_obs)
                )
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                mus[step] = mu
                sigmas[step] = sigma

                # TRY NOT TO MODIFY BEGIN: execute the game and log data.
                try:
                    next_obs, reward, next_done, infos = envs.step(action)
                except Exception:
                    action = action.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                    next_obs, reward, next_done, infos = envs.step(action)
                # # Bootstrapping on time outs - disabled for finite horizon tasks
                # if "time_outs" in infos:
                #     reward += args.gamma * torch.squeeze(
                #         value * infos["time_outs"].unsqueeze(1).to(device), 1
                #     )
                rewards[step] = reward.view(-1)
                dones[step] = next_done
                # TRY NOT TO MODIFY END:

                # Capture detailed logging information
                if "log" in infos:
                    log_data = infos["log"]

                    # Capture Episode Rewards
                    for key, value in log_data.items():
                        if key.startswith("Episode_Reward/"):
                            reward_name = key.replace("Episode_Reward/", "")
                            if reward_name not in reward_info_buffer:
                                reward_info_buffer[reward_name] = []
                            reward_info_buffer[reward_name].append(value)

                    # Capture Metrics
                    for key, value in log_data.items():
                        if key.startswith("Metrics/"):
                            metric_name = key.replace("Metrics/", "")
                            if metric_name not in metric_info_buffer:
                                metric_info_buffer[metric_name] = []
                            metric_info_buffer[metric_name].append(value)

                    # Capture Curriculum info
                    for key, value in log_data.items():
                        if key.startswith("Curriculum/"):
                            curriculum_name = key.replace("Curriculum/", "")
                            if curriculum_name not in curriculum_info_buffer:
                                curriculum_info_buffer[curriculum_name] = []
                            curriculum_info_buffer[curriculum_name].append(value)

                    # Capture Termination info
                    for key, value in log_data.items():
                        if key.startswith("Episode_Termination/"):
                            termination_name = key.replace("Episode_Termination/", "")
                            if termination_name not in termination_info_buffer:
                                termination_info_buffer[termination_name] = []
                            termination_info_buffer[termination_name].append(value)

                if args.log_video:
                    frame = next_obs[indices, : 3 * 32 * 32].reshape(-1, 3, 32, 32)
                    frame = (
                        torchvision.utils.make_grid(frame, nrow=3, scale_each=True)
                        * 255.0
                    )
                    video_frames.append(frame)

                # Update step progress bar
                step_pbar.update(1)
            step_speed = (args.num_steps * args.num_envs) / (time.time() - start_time)
            step_pbar.set_description(f"speed (sps) : {step_speed:3.1f}")

            # bootstrap value if not done
            next_value = inference_agent.get_value(next_obs).reshape(1, -1)
            returns = torch.zeros_like(rewards).to(device)

            advantage = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = values[t + 1]
                nextnonterminal = 1.0 - dones[t]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantage = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * advantage
                )
                returns[t] = advantage + values[t]
            advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape[1:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape[1:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_mus = mus.reshape((-1,) + envs.action_space.shape[1:])
        b_sigmas = sigmas.reshape((-1,) + envs.action_space.shape[1:])

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, newmu, newsigma = (
                    agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.inference_mode():
                    kl_mean = torch.mean(
                        torch.sum(
                            torch.log(newsigma / b_sigmas[mb_inds] + 1.0e-5)
                            + (
                                torch.square(b_sigmas[mb_inds])
                                + torch.square(b_mus[mb_inds] - newmu)
                            )
                            / (2 * torch.square(newsigma))
                            - 0.5,
                            dim=-1,
                        )
                    )
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

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
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = v_loss_max.mean()
                else:
                    v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()

            if args.target_kl is not None and args.adaptive_lr:
                new_lr = update_learning_rate_adaptive(
                    optimizer,
                    kl_mean.item(),
                    args.target_kl,
                    args.lr_multiplier,
                    min_lr=1e-7,
                    max_lr=1e-3,
                )

            # Update EMA weights after all update epochs
            if (
                args.use_ema
                and ema_agent is not None
                and global_step >= args.ema_start_step
            ):
                update_ema_weights(ema_agent, agent, args.ema_decay)

        # Log the learning rate change
        if global_step_burnin is not None and iteration % args.log_interval == 0:
            wandb.log(
                {"learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step
            )

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if global_step_burnin is not None and iteration % args.log_interval == 0:
            desc = f"gl_st={global_step:3.0F}, "
            with torch.no_grad():
                logs = {
                    "logprobs": b_logprobs.mean(),
                    "advantages": advantages.mean(),
                    "returns": returns.mean(),
                    "mu": b_mus.mean(),
                    "sigma": b_sigmas.mean(),
                    "values": values.mean(),
                    "grad_norm": grad_norm,
                    "explained_var": explained_var,
                    "kl_mean": kl_mean,
                    "metrics/action_max": b_actions.max().item(),
                    "metrics/action_min": b_actions.min().item(),
                    "metrics/obs_max": b_obs.max().item(),
                    "metrics/obs_min": b_obs.min().item(),
                }

                # Add reward terms
                for reward_name, reward_values in reward_info_buffer.items():
                    if len(reward_values) > 0:
                        logs[f"rewards/{reward_name}"] = (
                            torch.tensor(reward_values).float().mean()
                        )
                # Add metrics
                for metric_name, metric_values in metric_info_buffer.items():
                    if len(metric_values) > 0:
                        logs[f"metrics/{metric_name}"] = (
                            torch.tensor(metric_values).float().mean()
                        )
                # Add curriculum info
                for (
                    curriculum_name,
                    curriculum_values,
                ) in curriculum_info_buffer.items():
                    if len(curriculum_values) > 0:
                        logs[f"curriculum/{curriculum_name}"] = (
                            torch.tensor(curriculum_values).float().mean()
                        )
                # Add termination info
                for (
                    termination_name,
                    termination_values,
                ) in termination_info_buffer.items():
                    if len(termination_values) > 0:
                        logs[f"terminations/{termination_name}"] = (
                            torch.tensor(termination_values).float().mean()
                        )
                # Clear all buffers for next interval
                if metric_info_buffer.get("avg_step_reward", None):
                    avg_returns = (
                        torch.tensor(metric_info_buffer["avg_step_reward"])
                        .mean()
                        .item()
                    )
                if metric_info_buffer.get("episode_reward", None):
                    ep_ret = (
                        torch.tensor(metric_info_buffer["episode_reward"]).mean().item()
                    )
                if metric_info_buffer.get("max_episode_return", None):
                    max_ep_ret = (
                        torch.tensor(metric_info_buffer["max_episode_return"])
                        .mean()
                        .item()
                    )
                if metric_info_buffer.get("max_step_reward", None):
                    max_step_reward = (
                        torch.tensor(metric_info_buffer["max_step_reward"])
                        .mean()
                        .item()
                    )
                reward_info_buffer.clear()
                metric_info_buffer.clear()
                curriculum_info_buffer.clear()
                termination_info_buffer.clear()

                # # Check success rate and update avoid_penalty_weight if needed
                # if logs.get("metrics/success_rate", 0) > 0.50 and hasattr(
                #     envs.unwrapped, "avoid_penalty_weight"
                # ):
                #     envs.unwrapped.avoid_penalty_weight = (
                #         envs.unwrapped.cfg.avoid_penalty_weight
                #     )

            desc = (
                desc
                + f"ep_ret={ep_ret:3.1f}, max_ep_ret={max_ep_ret:3.1f}, step_rew={avg_returns:2.2f}, max_step_rew={max_step_reward:2.2f}"
            )
            iteration_pbar.set_description(desc)
            wandb.log(
                {
                    "speed": step_speed,
                    **logs,
                },
                step=global_step,
            )

            if args.log_video and len(video_frames) > 0:
                video_tensor = torch.stack(video_frames)
                wandb.log(
                    {
                        "obs_grid_video": wandb.Video(
                            video_tensor.detach().cpu().numpy().astype(np.uint8),
                            fps=25,
                            format="mp4",
                        )
                    },
                    step=global_step,
                )
                video_frames = []
                indices = torch.randperm(args.num_envs)[: min(9, args.num_envs)].to(
                    device
                )
        # save every checkpoint_interval steps
        if global_step_burnin is not None and iteration % args.checkpoint_interval == 0:
            # Choose which agent to save based on configuration
            agent_to_save = (
                ema_agent
                if (args.use_ema and args.use_ema_for_eval and ema_agent is not None)
                else agent
            )

            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{global_step}.pt")

            # Save both regular and EMA weights if EMA is enabled
            if args.use_ema and ema_agent is not None:
                checkpoint_data = {
                    "model_state_dict": agent.state_dict(),
                    "ema_model_state_dict": ema_agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "iteration": iteration,
                    "args": vars(args),
                }
            else:
                checkpoint_data = {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "iteration": iteration,
                    "args": vars(args),
                }
            torch.save(checkpoint_data, ckpt_path)

            mean_return = float(avg_returns)

            if mean_return > best_return:
                best_return = mean_return
                best_step = global_step
                best_ckpt = ckpt_path

        avg_returns, ep_ret, max_ep_ret, max_step_reward = 0.0, 0.0, 0.0, 0.0
        # Update iteration progress bar
        iteration_pbar.update(1)

    # Close progress bars
    step_pbar.close()
    iteration_pbar.close()

    envs.close()
    del envs

    # upload the best checkpoint as an Artifact
    if best_ckpt is not None:
        artifact = wandb.Artifact(
            name="ppo-best-checkpoint",
            type="model",
            description=f"Best PPO model at step {best_step} with return {best_return:.2f}",
        )
        artifact.add_file(best_ckpt)
        run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    try:
        if not args.log:
            os.environ["WANDB_MODE"] = "dryrun"
        main(args)
    except Exception as e:
        print("Exception:", e)
    finally:
        simulation_app.close()


# python scripts/torchrl/ppo_continuous_action.py --enable_cameras --headless --task Spot-Nav-Avoid-v0 --num_envs 128 --total_timesteps 10_000_000 --log_interval 1 --checkpoint_interval 1_000  --log --ent_coef 0.000001
