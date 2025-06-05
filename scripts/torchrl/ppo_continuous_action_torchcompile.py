# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import time
from collections import deque
from dataclasses import asdict
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from isaaclab.utils import configclass
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CNNPPOAgent, MLPPPOAgent
from utils import load_args, seed_everything, set_high_precision


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
    """device to use for training"""

    # Algorithm specific arguments

    """the id of the environment"""
    total_timesteps: int = 1_000_000
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
    agent_type: str = "MLPPPOAgent"

    compile: bool = True
    """whether to use torch.compile."""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile."""


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
        env = IsaacLabVecEnvWrapper(env)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def main(args):
    # def gae(next_obs, next_done, container):
    #     # bootstrap value if not done
    #     next_value = get_value(next_obs).reshape(-1)
    #     lastgaelam = 0
    #     nextnonterminals = container["dones"].float().unbind(0)
    #     vals = container["vals"]
    #     vals_unbind = vals.unbind(0)
    #     rewards = container["rewards"].unbind(0)

    #     advantages = []
    #     nextnonterminal = next_done.float()
    #     nextvalues = next_value
    #     for t in range(args.num_steps - 1, -1, -1):
    #         cur_val = vals_unbind[t]
    #         delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
    #         advantages.append(
    #             delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    #         )
    #         lastgaelam = advantages[-1]

    #         nextnonterminal = nextnonterminals[t]
    #         nextvalues = cur_val

    #     advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    #     container["returns"] = advantages + vals
    #     return container

    # 4) Fully tensorized GAE
    def gae(next_val, next_done, container):
        """
        Fully tensorized GAE via reverse-cumsum, using correct (1 - done) masking.
        Returns advantages and returns of shape [T, N].
        """
        rewards = container["rewards"]  # [T, N]
        values = container["vals"]  # [T, N]
        dones = container["dones"]  # [T, N]
        # Bootstrap V_T from next_obs
        with torch.no_grad():
            next_val = get_value(next_obs).view(1, -1)
        # Align next values for each timestep
        next_vals = torch.cat([values[1:], next_val], dim=0)  # [T, N]

        nonterm = 1.0 - torch.cat([dones[1:], next_done.view(1, -1)], dim=0)

        # δ_t = r_t + γ·V_{t+1}·nonterm_t – V_t
        deltas = rewards + args.gamma * next_vals * nonterm - values  # [T, N]
        advantages = deltas.clone()
        for t in range(args.num_steps - 2, -1, -1):
            advantages[t] += (
                args.gamma * args.gae_lambda * nonterm[t] * advantages[t + 1]
            )
        container.set_("advantages", advantages)
        container.set_("returns", advantages + values)
        return container

    def rollout(obs, done, avg_returns=[]):
        ts = []
        for step in range(args.num_steps):
            # ALGO LOGIC: action logic
            action, logprob, _, value = policy(obs=obs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, infos = step_func(action)

            if "episode" in infos:
                avg_returns.extend(infos["episode"]["r"])
                # desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
            td = fixed_td.clone()
            td.set_("obs", obs)
            td.set_("rewards", reward)
            td.set_("dones", done.float())
            td.set_("actions", action)
            td.set_("logprobs", logprob)
            td.set_("vals", value.squeeze(-1))

            ts.append(td)
            obs = next_obs
            done = next_done

        container = torch.stack(ts, 0)
        return next_obs, done.float(), container

    def update(obs, actions, logprobs, advantages, returns, vals):
        optimizer.zero_grad()
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
        logratio = newlogprob - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = vals + torch.clamp(
                newvalue - vals,
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        loss.backward()
        gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        return (
            approx_kl,
            v_loss.detach(),
            pg_loss.detach(),
            entropy_loss.detach(),
            old_approx_kl,
            clipfrac,
            gn,
        )

    update = tensordict.nn.TensorDictModule(
        update,
        in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
        out_keys=[
            "approx_kl",
            "v_loss",
            "pg_loss",
            "entropy_loss",
            "old_approx_kl",
            "clipfrac",
            "gn",
        ],
    )

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

    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )

    # env setup
    envs = make_isaaclab_env(
        args.task,
        args.device,
        args.num_envs,
        args.disable_fabric,
        args.capture_video,
    )()
    # TRY NOT TO MODIFY: seeding
    seed_everything(envs, args.seed, use_torch=True, torch_deterministic=True)
    set_high_precision()
    n_obs = int(np.prod(envs.observation_space.shape[1:]))
    n_act = int(np.prod(envs.action_space.shape[1:]))
    assert isinstance(envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    fixed_td = TensorDict(
        {
            "obs": torch.zeros(args.num_envs, n_obs, device=device),
            "rewards": torch.zeros(args.num_envs, device=device, dtype=torch.float32),
            "vals": torch.zeros(args.num_envs, device=device, dtype=torch.float32),
            "dones": torch.zeros(args.num_envs, device=device, dtype=torch.float32),
            "actions": torch.zeros(args.num_envs, n_act, device=device),
            "logprobs": torch.zeros(args.num_envs, device=device),
            "advantages": torch.zeros(args.num_envs, device=device),
            "returns": torch.zeros(args.num_envs, device=device),
        },
        batch_size=[args.num_envs],
    )

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        next_obs, reward, done, info = envs.step(action)
        return next_obs, reward, done, info

    ####### Agent #######
    if args.agent_type == "CNNPPOAgent":
        agent = CNNPPOAgent(n_obs, n_act).to(device)
        agent_inference = CNNPPOAgent(n_obs, n_act).to(device)
    else:
        agent = MLPPPOAgent(n_obs, n_act).to(device)
        agent_inference = MLPPPOAgent(n_obs, n_act).to(device)
    # Make a version of agent with detached params
    agent_inference.eval()
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    # ALGO Logic: Storage setup
    avg_returns = deque(maxlen=20)

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    if args.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    # next_obs_buf, _ = envs.reset(seed=args.seed)
    # next_obs = next_obs_buf["policy"]
    envs.seed(seed=args.seed)
    next_obs, _ = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    global_step_burnin = None
    start_time = None
    max_ep_ret = -float("inf")

    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(
            next_obs, next_done, avg_returns=avg_returns
        )
        global_step += container.numel()
        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(
                args.minibatch_size
            )
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break

        if global_step_burnin is not None and iteration % 10 == 0:
            speed = (global_step - global_step_burnin) / (time.time() - start_time)
            r = container["rewards"].mean()
            r_max = container["rewards"].max()
            max_ep_ret = max(max_ep_ret, r_max)
            avg_returns_t = torch.tensor(avg_returns).mean()

            with torch.no_grad():
                logs = {
                    "episode_return": np.array(avg_returns).mean(),
                    "logprobs": container["logprobs"].mean(),
                    "advantages": container["advantages"].mean(),
                    "returns": container["returns"].mean(),
                    "vals": container["vals"].mean(),
                    "gn": out["gn"].mean(),
                }

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"reward avg: {r:4.2f}, "
                f"reward max: {r_max:4.2f}, "
                f"returns: {avg_returns_t: 4.2f}, "
                f"ep_r_max: {max_ep_ret: 4.2f}"
            )
            wandb.log(
                {
                    "speed": speed,
                    "episode_return": avg_returns_t,
                    "r": r,
                    "r_max": r_max,
                    "lr": lr,
                    "ep_r_max": max_ep_ret,
                    **logs,
                },
                step=global_step,
            )

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    try:
        main(args)
    except Exception as e:
        print("Exception:", e)
    finally:
        simulation_app.close()
