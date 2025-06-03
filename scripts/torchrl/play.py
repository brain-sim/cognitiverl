# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaaclab.utils import configclass
from models import CNNPPOAgent

### TODO : Make play callable while training and after training.
### Solution - Use ManagerBasedRL or multi threading or multiprocessing to run train and eval.

### TODO : get the name of the video file from the environment.
### Solution - Use the environment's metadata to get the name of the video file (do it elegantly).


@configclass
class EnvArgs:
    task: str = "CognitiveRL-Nav-v2"
    """the id of the environment"""
    env_cfg_entry_point: str = "env_cfg_entry_point"
    """the entry point of the environment configuration"""
    num_envs: int = 64
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
    headless: bool = True
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
    """device to use for training"""

    checkpoint_path: str = "/home/user/cognitiverl/wandb/offline-run-20250602_194417-pnrq9ikt/files/checkpoints/ckpt_20480.pt"
    """path to the checkpoint to load"""
    num_eval_envs: int = 10
    """number of environments to run for evaluation/play."""
    num_eval_env_steps: int = 50
    """number of steps to run for evaluation/play."""
    agent = CNNPPOAgent


@configclass
class Args(ExperimentArgs, EnvArgs):
    pass


def launch_app(args):
    from argparse import Namespace

    app_launcher = AppLauncher(Namespace(**asdict(args)))
    return app_launcher.app


def get_args():
    exp_args = ExperimentArgs()
    env_args = EnvArgs()
    merged_args = {**asdict(exp_args), **asdict(env_args)}
    args = Args(**merged_args)
    return args


try:
    from isaaclab.app import AppLauncher

    args = get_args()
    simulation_app = launch_app(args)

except ImportError:
    raise ImportError("Isaac Lab is not installed. Please install it first.")


def make_isaaclab_env(
    task,
    device,
    num_envs,
    capture_video,
    disable_fabric,
    log_dir=None,
    video_length=200,
    *args,
    **kwargs,
):
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.torchrl import (
        IsaacLabRecordEpisodeStatistics,
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
        )
        env = IsaacLabRecordEpisodeStatistics(env)
        if capture_video and log_dir is not None:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env = IsaacLabVecEnvWrapper(env)
        return env

    return thunk


def main(args):
    run_name = f"{args.task}__{args.exp_name}"
    run = wandb.init(
        project="play",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )
    run_dir = run.dir

    eval_envs = make_isaaclab_env(
        args.task,
        args.device,
        args.num_eval_envs,
        args.capture_video,
        args.disable_fabric,
        log_dir=run_dir,
    )()

    n_obs = int(np.prod(eval_envs.observation_space.shape[1:]))
    n_act = int(np.prod(eval_envs.action_space.shape[1:]))
    assert isinstance(eval_envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    @torch.no_grad()
    def run_play(play_agent, mode="play"):
        play_agent.eval()
        # environment evaluation
        eval_envs.seed(seed=args.seed)
        obs, _ = eval_envs.reset()
        step, global_step = 0, 0
        done = torch.zeros(args.num_eval_envs, dtype=torch.bool)
        pbar = tqdm.tqdm(total=args.num_eval_env_steps)
        while step < args.num_eval_env_steps and not done.all():
            action, _, _, _ = play_agent.get_action_and_value(obs)
            obs, _, done, _ = eval_envs.step(action)
            step += 1
            global_step += args.num_eval_envs
            pbar.update(1)
            pbar.set_description(f"step: {step}")
        eval_envs.close()

        if args.capture_video:
            wandb.log(
                {
                    f"{mode}/video_episode": wandb.Video(
                        os.path.join(
                            run_dir,
                            "videos",
                            "play",
                            f"{eval_envs.video_name}.mp4",
                        ),
                        format="mp4",
                        fps=30,
                    )
                },
                step=global_step,
            )

    agent = args.agent(n_obs, n_act)
    agent.load_state_dict(torch.load(args.checkpoint_path))
    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    agent.to(device)
    agent.eval()
    run_play(agent)
    wandb.finish()


if __name__ == "__main__":
    try:
        main(args)
    except Exception as e:
        print("Exception:", e)
    finally:
        simulation_app.close()
