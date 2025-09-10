# needed to import for type hinting: Agent | list[Agent]
from __future__ import annotations

from typing import Any

import gymnasium as gym

try:
    import jax.dlpack
except ImportError:
    pass

# A generic mechanism for turning a JAX function into a PyTorch function.
from typing import Dict

import numpy as np
import torch
import torch.utils.dlpack
from gymnasium import Wrapper
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


def _apply_nan_to_num(
    x: torch.Tensor | Dict[str, torch.Tensor],
) -> torch.Tensor | Dict[str, torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    elif isinstance(x, dict):
        return {k: _apply_nan_to_num(v) for k, v in x.items()}
    else:
        return x


def get_video_name(env: Any) -> str | None:
    """
    Walk through a potentially‐wrapped Gym environment.
    If any layer defines an attribute (or property) named `_video_name`,
    return its value. Otherwise, return None.
    """
    current = env
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        # If this layer has `_video_name`, return its value
        if hasattr(current, "_video_name"):
            return getattr(current, "_video_name")

        # Unwrap one layer if possible
        if hasattr(current, "env"):
            current = current.env
        elif hasattr(current, "unwrapped"):
            current = current.unwrapped
        else:
            break

    return None


def j2t(x_jax):
    x_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))
    return x_torch


def t2j(x_torch):
    x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
    x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))
    return x_jax


def convert_dict_to_jax(obj):
    if torch.is_tensor(obj):
        return t2j(obj)
    elif isinstance(obj, dict):
        return {k: convert_dict_to_jax(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [convert_dict_to_jax(v) for v in obj]
        return type(obj)(converted)
    else:
        return obj


class IsaacLabVecEnvWrapper(Wrapper):
    """Wraps around Isaac Lab environment for RSL-RL library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "critic" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(
        self,
        env,
        clip_actions: float | None = None,
        action_bounds: float | None = None,
        use_jax: bool | False = False,
    ):
        super().__init__(env)
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(
            env.unwrapped, DirectRLEnv
        ):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.clip_actions = clip_actions
        self.action_bounds = action_bounds
        self.use_jax = use_jax
        print(f"Using JAX: {self.use_jax}")

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(
                self.unwrapped.single_observation_space["policy"]
            )
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim[
                "critic"
            ][0]
        elif (
            hasattr(self.unwrapped, "num_states")
            and "critic" in self.unwrapped.single_observation_space
        ):
            self.num_privileged_obs = gym.spaces.flatdim(
                self.unwrapped.single_observation_space["critic"]
            )
        else:
            self.num_privileged_obs = 0
        self.max_episode_steps = self.unwrapped.max_episode_length

        # modify the action space to the clip range
        self._modify_action_space()

        self._video_name = None
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        if self.use_jax:
            obs_dict = convert_dict_to_jax(obs_dict)
        obs_dict["policy"] = _apply_nan_to_num(obs_dict["policy"])
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    @property
    def video_name(self):
        return self._video_name

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self, random_start_init: bool = False) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # NOTE: decorrelate episode horizons like RSL‑RL
        if random_start_init:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_steps)
            )
        # return observations
        if self.use_jax:
            obs_dict = convert_dict_to_jax(obs_dict)
        obs_dict["policy"] = _apply_nan_to_num(obs_dict["policy"])
        return obs_dict["policy"], {"observations": obs_dict}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.use_jax:
            if isinstance(actions, jax.Array):
                actions = j2t(actions)
            elif isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        if self.action_bounds is not None:
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_bounds
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        if self.use_jax:
            obs_dict = convert_dict_to_jax(obs_dict)
            rew, terminated, truncated, dones = (
                t2j(rew),
                t2j(terminated),
                t2j(truncated),
                t2j(dones),
            )
            extras = convert_dict_to_jax(extras)

        obs = _apply_nan_to_num(obs_dict["policy"])
        # rew = rew.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
            extras["terminations"] = terminated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        self._video_name = get_video_name(self.env)
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
