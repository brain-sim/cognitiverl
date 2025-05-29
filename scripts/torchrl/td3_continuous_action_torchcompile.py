# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

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
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import seed_everything

### TODO : Batch size (global) and transition batch size should be different.
### The current code only works if they are both the same.


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
    buffer_size: int = 1_000_000
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

    cudagraphs: bool = True
    """use cudagraphs"""
    compile: bool = True
    """use torch.compile"""


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
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, env, exploration_noise=1, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mu = nn.Linear(256, n_act, device=device)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high[0] - env.action_space.low[0]) / 2.0,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high[0] + env.action_space.low[0]) / 2.0,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
        )
        self.register_buffer(
            "exploration_noise", torch.as_tensor(exploration_noise, device=device)
        )

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = self.fc_mu(obs).tanh()
        return obs * self.action_scale + self.action_bias

    def explore(self, obs):
        act = self(obs)
        return act + torch.randn_like(act).mul(
            self.action_scale * self.exploration_noise
        )


def main(args):
    run_name = (
        f"{args.task}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"
    )

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
    action_low = float(envs.action_space.high[0].max())
    action_high = float(envs.action_space.low[0].min())
    assert isinstance(envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    actor = Actor(
        env=envs,
        n_obs=n_obs,
        n_act=n_act,
        device=device,
        exploration_noise=args.exploration_noise,
    )
    actor_detach = Actor(
        env=envs,
        n_obs=n_obs,
        n_act=n_act,
        device=device,
        exploration_noise=args.exploration_noise,
    )
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    def get_params_qnet():
        qf1 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)
        qf2 = QNetwork(n_obs=n_obs, n_act=n_act, device=device)

        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target_params = qnet_params.data.clone()

        # discard params of net
        qnet = QNetwork(n_obs=n_obs, n_act=n_act, device="meta")
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target_params, qnet

    def get_params_actor(actor):
        target_actor = Actor(env=envs, device="meta", n_act=n_act, n_obs=n_obs)
        actor_params = from_module(actor).data
        target_actor_params = actor_params.clone()
        target_actor_params.to_module(target_actor)
        return actor_params, target_actor_params, target_actor

    qnet_params, qnet_target_params, qnet = get_params_qnet()
    actor_params, target_actor_params, target_actor = get_params_actor(actor)

    q_optimizer = optim.Adam(
        qnet_params.values(include_nested=True, leaves_only=True),
        lr=args.learning_rate,
        capturable=args.cudagraphs and not args.compile,
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()),
        lr=args.learning_rate,
        capturable=args.cudagraphs and not args.compile,
    )

    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.buffer_size, device=torch.device("cpu"))
    )

    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip
    action_scale = target_actor.action_scale

    def update_main(data):
        observations, next_observations, actions, rewards, dones = (
            data["observations"],
            data["next_observations"],
            data["actions"],
            data["rewards"],
            data["dones"],
        )
        clipped_noise = torch.randn_like(actions)
        clipped_noise = (
            clipped_noise.mul(policy_noise)
            .clamp(-noise_clip, noise_clip)
            .mul(action_scale)
        )

        next_state_actions = (target_actor(next_observations) + clipped_noise).clamp(
            action_low, action_high
        )

        qf_next_target = torch.vmap(batched_qf, (0, None, None))(
            qnet_target_params, next_observations, next_state_actions
        )
        min_qf_next_target = qf_next_target.min(0).values
        next_q_value = (
            rewards.flatten()
            + (~dones.flatten()).float() * args.gamma * min_qf_next_target.flatten()
        )

        qf_loss = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, observations, actions, next_q_value
        )
        qf_loss = qf_loss.sum(0)

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_pol(data):
        observations = data["observations"]
        actor_optimizer.zero_grad()
        with qnet_params.data[0].to_module(qnet):
            actor_loss = -qnet(observations, actor(observations)).mean()

        actor_loss.backward()
        actor_optimizer.step()
        return TensorDict(actor_loss=actor_loss.detach())

    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(args.batch_size)

    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(
            update_main,
            warmup=5,
        )
        update_pol = CudaGraphModule(
            update_pol,
            warmup=5,
        )
        policy = CudaGraphModule(policy)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""

    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        #     # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = torch.from_numpy(envs.action_space.sample()).float().to(device)
        else:
            actions = policy(obs=obs)

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
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
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
            out_main = update_main(data)
            if global_step % args.policy_frequency == 0:
                out_main.update(update_pol(data))

                # update the target networks
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target_params.lerp_(qnet_params.data, args.tau)
                target_actor_params.lerp_(actor_params.data, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "episode_return": torch.tensor(avg_returns).mean(),
                        "actor_loss": out_main["actor_loss"].mean(),
                        "qf_loss": out_main["qf_loss"].mean(),
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
        os.environ["WANDB_MODE"] = "dryrun"
        main(args)
    except Exception as e:
        print("Exception:", e)
    finally:
        simulation_app.close()
