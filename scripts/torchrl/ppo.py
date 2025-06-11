# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from collections import deque
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from isaaclab.utils import configclass
from torch.distributions.normal import Normal

try:
    from isaaclab.app import AppLauncher
except ImportError:
    raise ImportError("Isaac Lab is not installed. Please install it first.")


@configclass
class EnvArgs:
    task: str = "Leatherback-Nav-v2"
    """the id of the environment"""
    env_cfg_entry_point: str = "env_cfg_entry_point"
    """the entry point of the environment configuration"""
    num_envs: int = 512
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
    """device to use for training"""

    # Algorithm specific arguments

    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.01
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    measure_burnin: int = 3


@configclass
class Args(ExperimentArgs, EnvArgs):
    pass


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


def make_isaaclab_env(task, device, num_envs, capture_video, disable_fabric, **args):
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
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
        env = RslRlVecEnvWrapper(env)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def launch_app(args):
    from argparse import Namespace

    print(args)
    app_launcher = AppLauncher(Namespace(**asdict(args)))
    return app_launcher


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.observation_space.shape[1:]).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.observation_space.shape[1:]).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape[1:])), std=0.01),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.action_space.shape[1:]))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class CNN_Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        # Constants for the architecture
        self.img_channels = 3
        self.img_height = 32
        self.img_width = 32
        self.img_size = self.img_channels * self.img_height * self.img_width

        # Load pretrained MobileNetV3 Small
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # Set to eval mode BEFORE any forward pass
        self.backbone.eval()

        # Modify first layer to accept 32x32 input (original expects 224x224)
        self.backbone.features[0][0] = nn.Conv2d(
            3, 16, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Keep only the feature extractor
        self.backbone = nn.Sequential(*list(self.backbone.features))

        # Get feature size with batch size 2 to avoid batch norm issues
        dummy_input = torch.zeros(2, 3, 32, 32)
        with torch.no_grad():
            feature_size = self.backbone(dummy_input).flatten(1).shape[1]

        # Feature network after CNN
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(feature_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
        )

        # Critic (value) network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor mean network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape[1:])), std=0.01),
        )

        # Log standard deviation parameter
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.action_space.shape[1:]))
        )

    def extract_image(self, x):
        batch_size = x.shape[0]

        # Take only the first img_size elements from states (the image data)
        images = x[:, : self.img_size]

        # Reshape images to proper format: [B, C, H, W]
        images = images.reshape(
            batch_size, self.img_channels, self.img_height, self.img_width
        )

        # Process with MobileNetV3
        with torch.no_grad():
            img_features = self.backbone(images).flatten(1)

        return img_features

    def get_value(self, x):
        img_features = self.extract_image(x)
        features = self.feature_net(img_features)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        img_features = self.extract_image(x)
        features = self.feature_net(img_features)

        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(features),
        )


def main(args):
    run_name = f"{args.task}__{args.exp_name}__{args.seed}"

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    wandb.init(
        project="ppo_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)

    # env setup
    envs = make_isaaclab_env(
        args.task,
        args.device,
        args.num_envs,
        args.disable_fabric,
        args.capture_video,
    )()
    assert isinstance(envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    # agent = Agent(envs).to(device)
    agent = CNN_Agent(envs).to(device)
    agent = torch.compile(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Key variables
    # obs: (num_steps, num_envs, obs_dim), state of the environment (s)
    obs = torch.zeros((args.num_steps,) + envs.observation_space.shape).to(device)
    # actions: (num_steps, num_envs, action_dim), actions taken by the agent (a)
    actions = torch.zeros((args.num_steps,) + envs.action_space.shape).to(device)
    # logprobs: (num_steps, num_envs), score function log π(a|s;θ)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # rewards: (num_steps, num_envs), rewards received after taking each action (r)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # dones: (num_steps, num_envs), whether the episode is done (d)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # values: (num_steps, num_envs), value function v(s)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # avg_returns: deque(maxlen=20)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    # next_obs_buf, _ = envs.reset(seed=args.seed)
    # next_obs = next_obs_buf["policy"]
    envs.seed(seed=args.seed)
    next_obs, _ = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    max_ep_ret = -float("inf")
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    global_step_burnin = None
    start_time = None
    desc = ""

    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Main loop: iterate over the number of steps in the episode
        for step in tqdm.tqdm(range(0, args.num_steps)):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # next_obs_buf, reward, terminations, truncations, infos = envs.step(action)
            # next_obs = next_obs_buf["policy"]
            # next_done = torch.logical_or(terminations, truncations).float()
            next_obs, reward, next_done, infos = envs.step(action)
            rewards[step] = reward.view(-1)

            if step == args.num_steps - 1:
                x = (
                    next_obs[0, : 32 * 32 * 3].reshape(3, 32, 32).permute(1, 2, 0)
                )  # first env, last step image frame
                x_np = x.cpu().numpy()
                wandb.log(
                    {
                        "first env last frame": wandb.Image(
                            x_np, caption="Last observation"
                        )
                    },
                    step=global_step,
                )

            if "episode_reward" in infos:
                r = infos["episode_reward"]
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
                desc = f"global_step={global_step}, episodic_return={np.mean(avg_returns): 4.2f} (max={max_ep_ret: 4.2f})"

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape[1:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape[1:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
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
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if global_step_burnin is not None and iteration % 2 == 0:
            speed = (global_step - global_step_burnin) / (time.time() - start_time)
            pbar.set_description(f"speed: {speed: 4.1f} sps, " + desc)
            with torch.no_grad():
                logs = {
                    "episode_return": np.mean(avg_returns) if avg_returns else 0.0,
                    "logprobs": b_logprobs.mean().item(),
                    "advantages": advantages.mean().item(),
                    "returns": returns.mean().item(),
                    "values": values.mean().item(),
                    "gn": gn.item(),
                }
                if "episode_length" in infos:
                    logs["episode_length"] = infos["episode_length"]
                if "completion_percentage" in infos:
                    logs["completion_percentage"] = infos["completion_percentage"]
            wandb.log(
                {
                    "speed": speed,
                    **logs,
                },
                step=global_step,
            )

    envs.close()


if __name__ == "__main__":
    exp_args = ExperimentArgs()
    env_args = EnvArgs()
    merged_args = {**asdict(exp_args), **asdict(env_args)}
    args = Args(**merged_args)
    app_launcher = launch_app(env_args)
    main(args)
