# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
from dataclasses import asdict
from typing import Optional

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import tqdm
from isaaclab.utils import configclass
from termcolor import colored
from torch.amp import autocast

import wandb
from scripts.models import AGENT_LOOKUP_BY_ALGORITHM
from scripts.utils import (  # add load_args import
    EmpiricalNormalization,
    adjust_noise_scales,
    load_args,
    print_dict,
)

### TODO : Make play callable while training and after training.
### Solution - Use ManagerBasedRL or multi threading or multiprocessing to run train and eval.

### TODO : get the name of the video file from the environment.
### Solution - Use the environment's metadata to get the name of the video file (do it elegantly).


@configclass
class EnvArgs:
    task: str = "Spot-Nav-Avoid-v0"
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
    enable_cameras: bool = True
    """enable cameras to record sensor inputs."""
    clip_actions: Optional[float] = None
    """whether to clip the actions"""
    action_bounds: Optional[float] = None
    """the bounds of the action"""


@configclass
class ExperimentArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:0"
    """device to use for training"""

    checkpoint_path: str = "/home/chandramouli/cognitiverl/wandb/run-20250718_194601-qqo2ms79/files/checkpoints/ckpt_4000.pt"
    """path to the checkpoint to load"""
    num_eval_envs: int = 32
    """number of environments to run for evaluation/play."""
    num_eval_env_steps: int = 1000
    """number of steps to run for evaluation/play."""
    obs_type: str = "image"
    """the type of the observations"""
    algorithm: str = "fast_td3"
    """the algorithm to use"""
    amp: bool = True
    """whether to use amp"""
    amp_dtype: str = "bf16"
    """the dtype of the amp"""


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


def make_isaaclab_env(
    task,
    seed,
    device,
    num_envs,
    capture_video,
    disable_fabric,
    log_dir=None,
    video_length=200,
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
            play_mode=True,
        )
        if capture_video and log_dir is not None:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step % env.unwrapped.max_episode_length
                == 0,
                "video_length": video_length,
                # "fps": 30,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env = IsaacLabVecEnvWrapper(
            env,
            clip_actions=kwargs.get("clip_actions", None),
            action_bounds=kwargs.get("action_bounds", None),
        )
        return env

    return thunk


def main(args):
    print_dict(args)
    run_name = f"{args.task}__{args.exp_name}"
    run = wandb.init(
        project="play",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )
    run_dir = run.dir

    amp_enabled = args.amp and "cuda" in args.device and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if "cuda" in args.device and torch.cuda.is_available()
        else "mps"
        if "mps" in args.device and torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    checkpoint = torch.load(
        args.checkpoint_path, map_location=device, weights_only=False
    )
    print(checkpoint["args"]["clip_actions"])
    print(checkpoint["args"]["action_bounds"])
    eval_envs = make_isaaclab_env(
        args.task,
        args.seed,
        args.device,
        args.num_eval_envs,
        args.capture_video,
        args.disable_fabric,
        log_dir=run_dir,
        clip_actions=checkpoint["args"]["clip_actions"],
        action_bounds=1.0,  # checkpoint["args"]["action_bounds"],
        video_length=args.num_eval_env_steps,
    )()

    # Set camera eye position to (0, 0, 45) looking at origin
    print("📷 Setting camera eye position to (0, 0, 100)")
    # eval_envs.unwrapped.sim.set_camera_view(eye=[0, 0, 100], target=[0.0, 0.0, 0.0])
    n_obs = int(np.prod(eval_envs.num_obs))
    n_act = int(np.prod(eval_envs.num_actions))

    assert isinstance(eval_envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    agent_class = AGENT_LOOKUP_BY_ALGORITHM[args.algorithm][args.obs_type]
    if isinstance(agent_class, tuple) or isinstance(agent_class, list):
        actor, critic = agent_class
        if args.obs_type == "image":
            agent = actor(
                n_obs,
                n_act,
                num_envs=args.num_eval_envs,
                device=device,
                init_scale=checkpoint["args"]["init_scale"],
                img_size=checkpoint["args"]["img_size"],
            )
        else:
            agent = actor(
                n_obs,
                n_act,
                num_envs=args.num_eval_envs,
                device=device,
                init_scale=checkpoint["args"]["init_scale"],
                hidden_dims=checkpoint["args"]["actor_hidden_dims"],
            )
    else:
        actor_class = agent_class
        if args.obs_type == "image":
            agent = actor_class(n_obs, n_act, img_size=checkpoint["args"]["img_size"])
        else:
            agent = actor_class(n_obs, n_act)
        critic = None

    print(
        colored(
            f"[INFO] : Loading agent from {args.checkpoint_path}",
            "green",
            attrs=["bold"],
        )
    )

    if "actor_state_dict" in checkpoint.keys():
        state_dict = checkpoint["actor_state_dict"].copy()
        state_dict = adjust_noise_scales(state_dict, agent, args.num_eval_envs)
        agent.load_state_dict(state_dict)
    else:
        state_dict = checkpoint.copy()
        state_dict = adjust_noise_scales(state_dict, agent, args.num_eval_envs)
        agent.load_state_dict(state_dict)

    if "obs_normalizer_state" in checkpoint.keys():
        obs_normalizer = EmpiricalNormalization(n_obs, device=args.device)
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    else:
        obs_normalizer = nn.Identity()
    normalize_obs = obs_normalizer.forward
    agent.to(device)
    agent.eval()

    @torch.no_grad()
    @autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled)
    def run_play(play_agent, mode="play"):
        play_agent.eval()
        obs, _ = eval_envs.reset()
        step, global_step = 0, 0
        done = torch.zeros(args.num_eval_envs, dtype=torch.bool)
        # Initialize storage for observations
        obs_history = []

        # Create video directory
        video_dir = os.path.join(run_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)

        img_size = eval_envs.unwrapped.cfg.img_size

        pbar = tqdm.tqdm(total=args.num_eval_env_steps)
        while step < args.num_eval_env_steps and not done.all():
            # Store current observations
            obs_history.append(obs.clone().cpu().numpy())

            obs = normalize_obs(obs, update=False)
            if hasattr(play_agent, "get_action"):
                action = play_agent.get_action(obs)
            else:
                action = play_agent(obs)
            obs, _, done, _ = eval_envs.step(action)
            step += 1
            global_step += args.num_eval_envs
            pbar.update(1)
            pbar.set_description(f"step: {step}")

        # Save final observation
        obs_history.append(obs.clone().cpu().numpy())

        # Convert observations to videos for each environment
        obs_array = np.array(obs_history)  # Shape: (steps, n_envs, obs_dim)

        # Create videos for each environment
        for env_idx in range(args.num_eval_envs):
            # Extract observations for this environment
            env_obs = obs_array[
                :, env_idx, : img_size[0] * img_size[1] * img_size[2]
            ]  # Take RGB channels
            # Reshape to (steps, height, width, channels)
            env_frames = env_obs.reshape(
                -1, img_size[0], img_size[1], img_size[2]
            ).transpose(0, 2, 3, 1)

            # Convert from float [0,1] to uint8 [0,255] if needed
            if env_frames.max() <= 1.0:
                env_frames = (env_frames * 255).astype(np.uint8)
            else:
                env_frames = env_frames.astype(np.uint8)

            # No need to convert RGB to BGR for imageio (keep RGB format)

            # Create video using imageio
            video_path = os.path.join(video_dir, f"env_{env_idx}_fpp.mp4")

            # Save video with imageio
            imageio.mimwrite(video_path, env_frames, fps=10, quality=8)

            print(f"Saved video for environment {env_idx}: {video_path}")

        eval_envs.close()

    run_play(agent)
    wandb.finish()


if __name__ == "__main__":
    try:
        os.environ["WANDB_MODE"] = "dryrun"
        main(args)
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print("Exception:", e)
    finally:
        simulation_app.close()
