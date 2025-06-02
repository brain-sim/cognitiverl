# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import sys
import time
from collections import deque
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
from isaaclab.utils import configclass
from torchrl.data import LazyTensorStorage, ReplayBuffer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensordict import TensorDict
from utils import seed_everything


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
    """cuda:0 will be enabled by default"""

    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 25_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = int(25e3)
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    measure_burnin: int = 3

    # Agent config
    agent_type: str = "CNNTD3Agent"

    checkpoint_interval: int = 10000
    """environment steps between saving checkpoints."""
    play_interval: int = 10000
    """environment steps between playing evaluation episodes during training."""
    run_play: bool = False
    """whether to play evaluation episodes during training."""
    render_play: bool = False
    """whether to render episodes when playing during training."""
    render_best: bool = False
    """whether to render episodes when running the best model after training."""
    num_eval_episodes: int = 1
    """number of episodes to run for evaluation/play."""

    def __post_init__(self):
        self.buffer_size = min(self.buffer_size, self.total_timesteps)


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


def make_env(task, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(task, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(task)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def make_isaaclab_env(task, device, num_envs, capture_video, disable_fabric, **args):
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
            render_mode="rgb_array" if capture_video else None,
        )
        env = IsaacLabRecordEpisodeStatistics(env)
        env = IsaacLabVecEnvWrapper(env, clip_actions=1.0)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.fc1 = nn.Linear(
            n_obs + n_act,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, max_action, min_action):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, n_act)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (max_action - min_action) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (max_action + min_action) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def main(args):
    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(args.batch_size)

    def extend(transition):
        rb.extend(transition)

    def sample():
        return rb.sample(args.batch_size)

    run_name = f"{args.task}__{args.exp_name}__{args.seed}"

    wandb.init(
        project="td3_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )

    # env setup
    envs = make_isaaclab_env(
        args.task, args.device, args.num_envs, args.disable_fabric, args.capture_video
    )()
    # TRY NOT TO MODIFY: seeding
    seed_everything(envs, args.seed, use_torch=True, torch_deterministic=True)
    n_obs = int(np.prod(envs.observation_space.shape[1:]))
    n_act = int(np.prod(envs.action_space.shape[1:]))
    assert isinstance(envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    max_action = float(envs.action_space.high[0].max())
    min_action = float(envs.action_space.low[0].min())

    actor = Actor(n_obs, n_act, max_action, min_action).to(device)
    qf1 = QNetwork(n_obs, n_act).to(device)
    qf2 = QNetwork(n_obs, n_act).to(device)
    qf1_target = QNetwork(n_obs, n_act).to(device)
    qf2_target = QNetwork(n_obs, n_act).to(device)
    target_actor = Actor(n_obs, n_act, max_action, min_action).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    rb = ReplayBuffer(
        storage=LazyTensorStorage(args.buffer_size, device=torch.device("cpu"))
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    args.num_iterations = args.total_timesteps // args.num_envs
    args.learning_starts = args.learning_starts // args.num_envs
    pbar = tqdm.tqdm(range(args.num_iterations))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = torch.from_numpy(envs.action_space.sample()).float().to(device)
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.clamp(min_action, max_action)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            for r in infos["episode"]["r"]:
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.clone()
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=actions,
            rewards=rewards,
            terminations=infos["terminations"],
            dones=dones,
            batch_size=obs.shape[0],
            device=torch.device("cpu"),
        )
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        data = extend_and_sample(transition)
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = data.to(device)
            with torch.no_grad():
                clipped_noise = (
                    torch.randn_like(data["actions"], device=device) * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale

                next_state_actions = (
                    target_actor(data["next_observations"]) + clipped_noise
                ).clamp(min_action, max_action)
                qf1_next_target = qf1_target(
                    data["next_observations"], next_state_actions
                )
                qf2_next_target = qf2_target(
                    data["next_observations"], next_state_actions
                )
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data["rewards"].flatten() + (
                    1 - data["dones"].flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data["observations"], data["actions"]).view(-1)
            qf2_a_values = qf2(data["observations"], data["actions"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(
                    data["observations"], actor(data["observations"])
                ).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if (global_step % 100 == 0) and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "episode_return": torch.tensor(avg_returns).mean(),
                        "actor_loss": actor_loss.mean(),
                        "qf_loss": qf_loss.mean(),
                    }
                wandb.log(
                    {
                        "speed": speed,
                        **logs,
                    },
                    step=global_step,
                )

    envs.close()


if __name__ == "__main__":
    try:
        main(get_args())
    except Exception as e:
        print("Exception:", e)
    finally:
        simulation_app.close()
