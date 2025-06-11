# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaaclab.utils import configclass
from isaaclab.utils.dict import print_dict
from models import CNNPPOAgent, MLPPPOAgent
from utils import load_args, seed_everything


@configclass
class EnvArgs:
    task: str = "Leatherback-Nav-v0"
    """the id of the environment"""
    env_cfg_entry_point: str = "env_cfg_entry_point"
    """the entry point of the environment configuration"""
    num_envs: int = 512
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

    # Agent config
    agent_type: str = "CNNPPOAgent"

    checkpoint_interval: int = 10
    """environment steps between saving checkpoints."""
    # play_interval: int = 3
    # """environment steps between playing evaluation episodes during training."""
    # run_play: bool = True
    # """whether to play evaluation episodes during training."""
    # run_best: bool = True
    # """whether to run the best model after training."""
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
        print_dict({"max_episode_steps": env.unwrapped.max_episode_length}, nesting=4)
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


def main(args):
    run_name = f"{args.task}__{args.exp_name}__{args.seed}"

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # initialize wandb run
    run = wandb.init(
        project="ppo_continuous_action",
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
    assert isinstance(envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    if args.agent_type == "CNNPPOAgent":
        agent = CNNPPOAgent(n_obs, n_act).to(device)
    else:
        agent = MLPPPOAgent(n_obs, n_act).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps,) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps,) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0

    next_obs, _ = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    max_ep_ret = -float("inf")
    max_ep_reward = -float("inf")
    success_rates, avg_reward_per_step, avg_returns = [], [], []
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    global_step_burnin = None
    start_time = None
    desc = ""

    if args.log_video:
        video_frames = []
        # Randomly choose minimum of 9 environments or all environments
        indices = torch.randperm(args.num_envs)[: min(9, args.num_envs)].to(device)

    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
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
            if "episode" in infos:
                for r in infos["episode"]["r"]:
                    max_ep_ret = max(max_ep_ret, r)
                    avg_returns.append(r)
                for r in infos["episode"]["reward_max"]:
                    max_ep_reward = max(max_ep_reward, r)
                    avg_reward_per_step.append(r)
            if "success_rate" in infos:
                success_rates.append(infos["success_rate"])
            if args.log_video:
                frame = next_obs[indices, : 3 * 32 * 32].reshape(-1, 3, 32, 32)
                frame = (
                    torchvision.utils.make_grid(frame, nrow=3, scale_each=True) * 255.0
                )
                video_frames.append(frame)
            rewards[step] = reward.view(-1)

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
                grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if global_step_burnin is not None and iteration % args.log_interval == 0:
            speed = (global_step - global_step_burnin) / (time.time() - start_time)
            desc = f"gl_st={global_step:3.0F}, "
            with torch.no_grad():
                logs = {
                    "logprobs": b_logprobs.mean(),
                    "advantages": advantages.mean(),
                    "values": values.mean(),
                    "grad_norm": grad_norm,
                    "explained_var": explained_var,
                    "old_approx_kl": old_approx_kl,
                    "approx_kl": approx_kl,
                }
                if len(avg_returns) > 0:
                    logs["avg_episode_return"] = torch.tensor(avg_returns).mean()
                    logs["max_episode_return"] = max_ep_ret
                    desc += f"ep_ret : avg={torch.tensor(avg_returns).mean():.2f}, max={max_ep_ret:.2f}, "
                if len(avg_reward_per_step) > 0:
                    logs["avg_step_reward"] = torch.tensor(avg_reward_per_step).mean()
                    logs["max_step_reward"] = (
                        torch.tensor(avg_reward_per_step).max().item()
                    )
                    desc += f"rew_per_step : avg={torch.tensor(avg_reward_per_step).mean():.2f}, max={max_ep_reward:.2f}"
                if len(success_rates) > 0:
                    logs["success_rate"] = torch.tensor(success_rates).mean()
                success_rates, avg_reward_per_step, avg_returns = [], [], []
            pbar.set_description(f"spd(sps): {speed:3.1f}, " + desc)
            wandb.log(
                {
                    "speed": speed,
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
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_{global_step}.pt")
            torch.save(agent.state_dict(), ckpt_path)

            mean_return = float(torch.tensor(avg_returns).mean().nan_to_num())
            mean_return += 1e-9

            if mean_return > best_return:
                best_return = mean_return
                best_step = global_step
                best_ckpt = ckpt_path

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
