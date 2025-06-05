# multigpu_dist_ppo_ddp.py

import os
import sys
import time
from collections import deque
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import torch.distributed as dist  # [DDP]
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb

# Ensure Isaac Lab's path is on sys.path so that `AppLauncher` resolves
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaaclab.utils import configclass
from isaaclab.utils.dict import print_dict
from models import CNNPPOAgent, MLPPPOAgent
from utils import load_args, seed_everything  # add load_args import

# =============================================================================
# 1. CONFIG CLASSES (copied/merged from original script)
# =============================================================================


@configclass
class EnvArgs:
    task: str = "CognitiveRL-Nav-v2"
    """the id of the environment."""
    env_cfg_entry_point: str = "env_cfg_entry_point"
    """the entry point for the environment config."""
    num_envs: int = 64
    """number of parallel environments per GPU."""
    seed: int = 1
    """base seed (offset by global rank)."""
    capture_video: bool = False
    """whether to capture videos (only on rank 0)."""
    video: bool = False
    """record videos during training."""
    video_length: int = 200
    """length of recorded video (in steps)."""
    video_interval: int = 2000
    """interval between video recordings (in steps)."""
    disable_fabric: bool = False
    """disable fabric, use USD I/O."""
    distributed: bool = True
    """run training with DDP (multi‐GPU)."""
    headless: bool = True
    """run in headless mode."""
    enable_cameras: bool = True
    """enable sensors/cameras in environment."""


@configclass
class ExperimentArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """name of this experiment (derived from filename)."""
    torch_deterministic: bool = True
    """if True, sets cudnn.deterministic=True for reproducibility."""
    device: str = "cuda"
    """base device string; actual device is cuda:LOCAL_RANK."""
    total_timesteps: int = 25_000
    """total timesteps per GPU."""
    learning_rate: float = 3e-4
    """optimizer learning rate."""
    num_steps: int = 64
    """steps per environment per rollout."""
    anneal_lr: bool = True
    """anneal learning rate linearly to zero."""
    gamma: float = 0.99
    """discount factor gamma."""
    gae_lambda: float = 0.95
    """lambda for Generalized Advantage Estimation."""
    num_minibatches: int = 8
    """number of minibatches per update."""
    update_epochs: int = 10
    """K epochs to update policy/value."""
    norm_adv: bool = False
    """normalize advantages if True."""
    clip_coef: float = 0.2
    """PPO clip coefficient."""
    clip_vloss: bool = True
    """use clipped value loss."""
    ent_coef: float = 0.0
    """entropy coefficient."""
    vf_coef: float = 1.0
    """value function coefficient."""
    max_grad_norm: float = 1.0
    """max gradient norm for clipping."""
    target_kl: float = 0.01
    """stop early if approximate KL > target."""
    batch_size: int = 0
    """(computed) batch size per GPU = num_envs * num_steps."""
    minibatch_size: int = 0
    """(computed) minibatch size = batch_size // num_minibatches."""
    num_iterations: int = 0
    """(computed) iterations per GPU = total_timesteps // batch_size."""
    measure_burnin: int = 3
    """iterations before measuring speed (for logging)."""
    agent_type: str = "CNNPPOAgent"
    """either 'CNNPPOAgent' or 'MLPPPOAgent'."""
    checkpoint_interval: int = 1
    """iterations between saving checkpoints (per GPU)."""
    play_interval: int = 3
    """iterations between evaluation episodes."""
    run_play: bool = True
    """whether to run eval episodes during training."""
    run_best: bool = True
    """whether to run best model after training."""
    num_eval_envs: int = 3
    """number of envs for evaluation/play."""
    num_eval_env_steps: int = 200
    """steps per eval/play episode."""


@configclass
class Args(ExperimentArgs, EnvArgs):
    """Combined arguments (inherits all fields)."""

    pass


# =============================================================================
# 2. DDP SETUP & CLEANUP
# =============================================================================


def ddp_setup():
    """
    Initialize the default process group for DistributedDataParallel.
    Expects the following environment variables (set by torchrun/torch.distributed.launch):
      - WORLD_SIZE: total number of processes (GPUs)
      - RANK: global rank (0..WORLD_SIZE-1)
      - LOCAL_RANK: local GPU index on this node (0..gpus_per_node-1)
      - MASTER_ADDR, MASTER_PORT must also be set externally (torchrun does this automatically).
    Uses backend='nccl' for GPU‐to‐GPU communication.
    """
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )


def ddp_cleanup():
    """Destroy the process group."""
    dist.destroy_process_group()


# =============================================================================
# 3. APP LAUNCHER (original script had this)
# =============================================================================


def launch_app(args):
    """
    Original launch_app: wraps Args into argparse.Namespace for Isaac Lab's AppLauncher.
    Returns the simulation application handle.
    """
    from argparse import Namespace

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(Namespace(**asdict(args)))
    return app_launcher.app


def get_args():
    return load_args(Args)


# =============================================================================
# 4. ENVIRONMENT CREATION (unchanged, except rank‐aware video capture)
# =============================================================================


def make_env(env_id, idx, capture_video, run_name, gamma):
    """
    Create one Gym environment for a single process (per‐GPU).
    If capture_video=True and idx==0, record video to videos/{run_name}.
    idx is the index of the environment (0..num_envs-1).
    """

    def thunk():
        if capture_video and idx == 0:
            # only record from the first sub‐env on rank 0
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
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
    device_str,
    num_envs,
    capture_video,
    disable_fabric,
    log_dir=None,
    video_length=200,
    *args,
    **kwargs,
):
    """
    Create Isaac Lab environments wrapped for TorchRL.
    Each process (per‐GPU) calls this to spawn `num_envs` parallel envs.
    """
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.torchrl import (
        IsaacLabRecordEpisodeStatistics,
        IsaacLabVecEnvWrapper,
    )
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import cognitiverl.tasks  # noqa: F401

    def thunk():
        cfg = parse_env_cfg(
            task, device_str, num_envs=num_envs, use_fabric=not disable_fabric
        )
        env = gym.make(
            task,
            cfg=cfg,
            render_mode="rgb_array"
            if (capture_video and log_dir is not None)
            else None,
        )
        env = IsaacLabRecordEpisodeStatistics(env)
        env = IsaacLabVecEnvWrapper(env, clip_actions=1.0)
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


# =============================================================================
# 5. MAIN WORKER FOR EACH GPU (DDP or single‐GPU)
# =============================================================================


def main_worker(local_rank, args):
    """
    Entry point for each spawned process (one per GPU).
      - local_rank:  index of GPU on this node (0..gpus_per_node-1)
      - args:        merged Args dataclass
    This sets up:
      - Device binding (cuda:local_rank)
      - DDP initialization (if args.distributed)
      - Per‐GPU environment creation, model, optimizer, training loop
      - Rank‐aware logging/checkpointing via wandb and torch.save
    """

    # -------------------- A. DEVICE & DDP INIT --------------------
    torch.cuda.set_device(
        local_rank
    )  #  [oai_citation:8‡PyTorch Forums](https://discuss.pytorch.org/t/multi-gpu-training-on-single-node-with-distributeddataparallel/92557?utm_source=chatgpt.com) [oai_citation:9‡Medium](https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-1-single-gpu-example-d682c15217a8?utm_source=chatgpt.com)
    device = torch.device(f"cuda:{local_rank}")

    if args.distributed:
        ddp_setup()  #  [oai_citation:10‡PyTorch Documentation](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html?utm_source=chatgpt.com) [oai_citation:11‡GitHub](https://github.com/harveyp123/Pytorch-DDP-Example?utm_source=chatgpt.com)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # -------------------- B. SEEDING --------------------
    global_seed = args.seed + rank
    seed_everything(
        global_seed, use_torch=True, torch_deterministic=args.torch_deterministic
    )  #  [oai_citation:12‡GitHub](https://github.com/harveyp123/Pytorch-DDP-Example?utm_source=chatgpt.com) [oai_citation:13‡PyTorch Forums](https://discuss.pytorch.org/t/multi-gpu-training-on-single-node-with-distributeddataparallel/92557?utm_source=chatgpt.com)

    # -------------------- C. RANK 0 LOGGING & CHECKPOINTS --------------------
    if rank == 0:
        # Launch Isaac Lab simulator app (only rank 0)
        try:
            simulation_app = launch_app(args)  # integrate original launch_app
        except Exception as e:
            raise RuntimeError(f"Failed to launch simulation app: {e}")

        wandb.init(
            project="ppo_continuous_action_ddp",
            name=f"{os.path.splitext(os.path.basename(__file__))[0]}_rank{rank}",
            config=vars(args),
            save_code=True,
        )
        run = wandb.run
        run_dir = run.dir
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        simulation_app = None
        run = None
        ckpt_dir = None

    # -------------------- D. COMPUTE PER‐GPU BATCH SIZES --------------------
    args.batch_size = int(args.num_envs * args.num_steps)  # per GPU
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # per GPU
    args.num_iterations = args.total_timesteps // args.batch_size  # per GPU

    # -------------------- E. ENVIRONMENT SETUP --------------------
    # Only rank 0 captures video (others skip to reduce I/O contention)
    envs = make_isaaclab_env(
        args.task,
        f"cuda:{local_rank}",
        args.num_envs,
        args.capture_video and (rank == 0),
        args.disable_fabric,
        log_dir=run.dir if rank == 0 else None,
        video_length=args.video_length,
    )()

    n_obs = int(np.prod(envs.observation_space.shape[1:]))
    n_act = int(np.prod(envs.action_space.shape[1:]))
    assert isinstance(envs.action_space, gym.spaces.Box), (
        "Only continuous action spaces are supported."
    )

    # -------------------- F. AGENT & OPTIMIZER --------------------
    if args.agent_type == "CNNPPOAgent":
        local_agent = CNNPPOAgent(n_obs, n_act).to(device)
    else:
        local_agent = MLPPPOAgent(n_obs, n_act).to(device)

    if args.distributed:
        # If using BatchNorm in the network, convert to SyncBatchNorm first:
        # local_agent = nn.SyncBatchNorm.convert_sync_batchnorm(local_agent)
        local_agent = nn.parallel.DistributedDataParallel(
            local_agent, device_ids=[local_rank]
        )  #  [oai_citation:14‡PyTorch Documentation](https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html?utm_source=chatgpt.com) [oai_citation:15‡PyTorch Forums](https://discuss.pytorch.org/t/multi-gpu-training-on-single-node-with-distributeddataparallel/92557?utm_source=chatgpt.com)

    optimizer = optim.Adam(local_agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # -------------------- G. STORAGE BUFFERS --------------------
    obs = torch.zeros((args.num_steps,) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps,) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # -------------------- H. INITIALIZE ROLLOUT --------------------
    global_step = 0
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)
    max_ep_ret = -float("inf")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1), disable=(rank != 0))
    global_step_burnin = None
    start_time = None
    desc = ""

    # =============================================================================
    # 6. MAIN TRAINING LOOP (PER GPU)
    # =============================================================================
    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # ---- 1) Anneal learning rate if requested ----
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lrnow

        # ---- 2) Collect rollout: num_steps × num_envs samples ----
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if args.distributed:
                    action, logprob, _, value = local_agent.module.get_action_and_value(
                        next_obs
                    )
                else:
                    action, logprob, _, value = local_agent.get_action_and_value(
                        next_obs
                    )
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Step environments
            next_obs_np, reward_np, next_done_np, infos = envs.step(
                action.cpu().numpy()
            )
            next_obs = torch.tensor(next_obs_np, device=device, dtype=torch.float32)
            rewards[step] = torch.tensor(
                reward_np, device=device, dtype=torch.float32
            ).view(-1)
            next_done = torch.tensor(
                next_done_np, device=device, dtype=torch.float32
            ).view(-1)

            # Record episode returns for rank‐0 logging
            if "episode" in infos and rank == 0:
                for r in infos["episode"]["r"]:
                    max_ep_ret = max(max_ep_ret, r)
                    avg_returns.append(r)
                desc = (
                    f"global_step={global_step}, "
                    f"episodic_return={torch.tensor(avg_returns).mean():4.2f} "
                    f"(max={max_ep_ret:4.2f})"
                )

        # ---- 3) Compute advantages & returns ----
        with torch.no_grad():
            if args.distributed:
                next_value = local_agent.module.get_value(next_obs).reshape(1, -1)
            else:
                next_value = local_agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                )
            returns = advantages + values

        # ---- 4) Flatten the batch per GPU ----
        b_obs = obs.reshape((-1,) + envs.observation_space.shape[1:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape[1:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ---- 5) Optimize policy & value networks ----
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                obs_batch = b_obs[mb_inds].to(device)
                actions_batch = b_actions[mb_inds].to(device)
                old_logprobs_batch = b_logprobs[mb_inds].to(device)
                advs_batch = b_advantages[mb_inds].to(device)
                ret_batch = b_returns[mb_inds].to(device)
                val_batch = b_values[mb_inds].to(device)

                if args.distributed:
                    _, newlogprob, entropy, newvalue = (
                        local_agent.module.get_action_and_value(
                            obs_batch, actions_batch
                        )
                    )
                else:
                    _, newlogprob, entropy, newvalue = local_agent.get_action_and_value(
                        obs_batch, actions_batch
                    )
                newvalue = newvalue.view(-1)

                # Compute policy loss
                logratio = newlogprob - old_logprobs_batch
                ratio = logratio.exp()
                pg_loss1 = -advs_batch * ratio
                pg_loss2 = -advs_batch * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Compute value loss
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - ret_batch) ** 2
                    v_clipped = val_batch + torch.clamp(
                        newvalue - val_batch, -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - ret_batch) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - ret_batch) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(local_agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # ---- 6) Logging (only rank 0) ----
        if rank == 0 and global_step_burnin is not None and iteration % 10 == 0:
            speed = (global_step - global_step_burnin) / (time.time() - start_time)
            pbar.set_description(f"speed: {speed:4.1f} sps, " + desc)
            with torch.no_grad():
                logs = {
                    "episode_return": np.array(avg_returns).mean(),
                    "loss": loss.item(),
                    "speed": speed,
                }
            wandb.log(logs, step=global_step)

        # ---- 7) Checkpointing (only rank 0) ----
        if (
            rank == 0
            and global_step_burnin is not None
            and iteration % args.checkpoint_interval == 0
        ):
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{global_step}.pt")
            # Save underlying module's state_dict if distributed
            to_save = (
                local_agent.module.state_dict()
                if args.distributed
                else local_agent.state_dict()
            )
            torch.save(to_save, ckpt_path)

    # =============================================================================
    # 7. CLEANUP
    # =============================================================================
    envs.close()
    if rank == 0 and simulation_app is not None:
        simulation_app.close()
    if args.distributed:
        ddp_cleanup()  #  [oai_citation:16‡PyTorch Forums](https://discuss.pytorch.org/t/multi-gpu-training-on-single-node-with-distributeddataparallel/92557?utm_source=chatgpt.com) [oai_citation:17‡Medium](https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-1-single-gpu-example-d682c15217a8?utm_source=chatgpt.com)


# =============================================================================
# 8. MAIN ENTRY POINT
# =============================================================================


def main():
    """
    Main entry: parse args, decide DDP vs single‐GPU, and spawn processes if needed.
    """
    # 1) Get default args from dataclasses (can override via command‐line flags)
    args = get_args()
    os.environ["WANDB_MODE"] = "dryrun"

    # 2) If distributed is requested, expect torchrun to set WORLD_SIZE, RANK, LOCAL_RANK
    if args.distributed:
        # Launch one process per GPU: torchrun will already spawn processes,
        # so simply read LOCAL_RANK and call main_worker.
        local_rank = int(os.environ["LOCAL_RANK"])
        main_worker(local_rank, args)
    else:
        # Single‐GPU fallback: set rank/world_size to 1, local_rank=0
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        main_worker(0, args)


if __name__ == "__main__":
    # When using torchrun, MASTER_ADDR and MASTER_PORT are set automatically.
    main()
