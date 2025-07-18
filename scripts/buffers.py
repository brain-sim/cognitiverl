import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage


class TorchRLInfoLogger:
    """Ultra-efficient logger using TensorDict and TorchRL's native operations."""

    def __init__(self, device="cpu", buffer_size=1000):
        self.device = torch.device("cpu")  # Force CPU device for logger
        self.buffer_size = buffer_size

        # Use TensorDict for efficient storage and operations
        self.info_storage = LazyTensorStorage(buffer_size, device=self.device)

        # Single TensorDict to store all different log types including transition metrics
        self.log_buffer = TensorDict({}, batch_size=[buffer_size], device=self.device)
        self.buffer_idx = 0
        self.total_logged = 0

    def update(
        self, infos, obs_max=None, obs_min=None, action_max=None, action_min=None
    ):
        """Update buffer with new info and transition metrics using TensorDict's efficient operations."""
        # Get current buffer index
        idx = self.buffer_idx % self.buffer_size

        # Update transition metrics if provided - ensure they're on CPU
        if obs_max is not None:
            if "obs_max" not in self.log_buffer:
                self.log_buffer["obs_max"] = torch.full(
                    (self.buffer_size,),
                    float("-inf"),
                    device=self.device,
                    dtype=torch.float32,
                )
            # Convert to CPU if needed
            obs_max_cpu = (
                float(obs_max)
                if isinstance(obs_max, (int, float))
                else obs_max.cpu().item()
            )
            self.log_buffer["obs_max"][idx] = obs_max_cpu

        if obs_min is not None:
            if "obs_min" not in self.log_buffer:
                self.log_buffer["obs_min"] = torch.full(
                    (self.buffer_size,),
                    float("inf"),
                    device=self.device,
                    dtype=torch.float32,
                )
            # Convert to CPU if needed
            obs_min_cpu = (
                float(obs_min)
                if isinstance(obs_min, (int, float))
                else obs_min.cpu().item()
            )
            self.log_buffer["obs_min"][idx] = obs_min_cpu

        if action_max is not None:
            if "action_max" not in self.log_buffer:
                self.log_buffer["action_max"] = torch.full(
                    (self.buffer_size,),
                    float("-inf"),
                    device=self.device,
                    dtype=torch.float32,
                )
            # Convert to CPU if needed
            action_max_cpu = (
                float(action_max)
                if isinstance(action_max, (int, float))
                else action_max.cpu().item()
            )
            self.log_buffer["action_max"][idx] = action_max_cpu

        if action_min is not None:
            if "action_min" not in self.log_buffer:
                self.log_buffer["action_min"] = torch.full(
                    (self.buffer_size,),
                    float("inf"),
                    device=self.device,
                    dtype=torch.float32,
                )
            # Convert to CPU if needed
            action_min_cpu = (
                float(action_min)
                if isinstance(action_min, (int, float))
                else action_min.cpu().item()
            )
            self.log_buffer["action_min"][idx] = action_min_cpu

        # Process info logs
        if "log" in infos:
            log_data = infos["log"]

            # Process all log categories in one pass
            for key, value in log_data.items():
                # Convert to scalar and ensure it's on CPU
                if torch.is_tensor(value):
                    scalar_value = value.cpu().item()
                else:
                    scalar_value = float(value)

                # Initialize key in buffer if it doesn't exist
                if key not in self.log_buffer:
                    self.log_buffer[key] = torch.zeros(
                        self.buffer_size, device=self.device, dtype=torch.float32
                    )

                # Store the value
                self.log_buffer[key][idx] = scalar_value

        self.buffer_idx += 1
        self.total_logged += 1

    def get_averaged_logs(self):
        """Get averaged logs using TensorDict's efficient operations."""
        if self.total_logged == 0:
            return {}

        # Calculate how many valid entries we have
        valid_count = min(self.total_logged, self.buffer_size)

        # Use TensorDict's apply method for efficient batch operations
        def compute_mean(tensor):
            if valid_count <= self.buffer_size:
                return tensor[:valid_count].mean()
            else:
                # For circular buffer, average all entries
                return tensor.mean()

        def compute_max(tensor):
            if valid_count <= self.buffer_size:
                return tensor[:valid_count].max()
            else:
                return tensor.max()

        def compute_min(tensor):
            if valid_count <= self.buffer_size:
                return tensor[:valid_count].min()
            else:
                return tensor.min()

        averaged_logs = {}

        # Apply efficient averaging to all stored logs
        for key, values in self.log_buffer.items():
            if len(values.shape) > 0:  # Only process if we have data
                # Handle transition metrics specially (use max/min instead of mean)
                if key in ["action_max", "obs_max"]:
                    avg_value = compute_max(values)
                elif key in ["action_min", "obs_min"]:
                    avg_value = compute_min(values)
                else:
                    avg_value = compute_mean(values)

                # Categorize based on key prefix or special metrics
                if key in ["action_max", "action_min", "obs_max", "obs_min"]:
                    averaged_logs[f"metrics/{key}"] = avg_value
                elif key.startswith("Episode_Reward/"):
                    clean_key = key.replace("Episode_Reward/", "")
                    averaged_logs[f"rewards/{clean_key}"] = avg_value
                elif key.startswith("Metrics/"):
                    clean_key = key.replace("Metrics/", "")
                    averaged_logs[f"metrics/{clean_key}"] = avg_value
                elif key.startswith("Curriculum/"):
                    clean_key = key.replace("Curriculum/", "")
                    averaged_logs[f"curriculum/{clean_key}"] = avg_value
                elif key.startswith("Episode_Termination/"):
                    clean_key = key.replace("Episode_Termination/", "")
                    averaged_logs[f"terminations/{clean_key}"] = avg_value

        return averaged_logs

    def reset(self):
        """Reset buffers efficiently using TensorDict operations."""
        # Reset by clearing the TensorDict - very efficient
        self.log_buffer = TensorDict(
            {}, batch_size=[self.buffer_size], device=self.device
        )

        self.buffer_idx = 0
        self.total_logged = 0


class BaseReplayBuffer(nn.Module):
    """Base class for replay buffers that stores transitions in a circular buffer.

    Supports asymmetric observations and playground mode for memory efficiency.
    """

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        device=None,
    ):
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.asymmetric_obs = asymmetric_obs
        self.playground_mode = playground_mode and asymmetric_obs
        self.device = device

        # Initialize storage tensors
        self.observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (n_env, buffer_size, n_act),
            device=device,
            dtype=torch.float,
        )
        self.rewards = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.terminations = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.time_outs = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.next_observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )

        if asymmetric_obs:
            if self.playground_mode:
                # Only store the privileged part of observations (n_critic_obs - n_obs)
                self.privileged_obs_size = n_critic_obs - n_obs
                self.privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
                self.next_privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
            else:
                # Store full critic observations
                self.critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
                self.next_critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )

        self.ptr = 0

    @torch.no_grad()
    def extend(self, tensor_dict: TensorDict):
        """Add new transitions to the buffer."""
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["rewards"]
        dones = tensor_dict["dones"]
        terminations = tensor_dict["terminations"]
        time_outs = tensor_dict["time_outs"]
        next_observations = tensor_dict["next_observations"]

        # Validate inputs
        for k, v in tensor_dict.items():
            if v.isnan().any():
                raise ValueError(f"{k} nan")
            if v.isinf().any():
                raise ValueError(f"{k} inf")

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.terminations[:, ptr] = terminations
        self.time_outs[:, ptr] = time_outs
        self.next_observations[:, ptr] = next_observations

        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next_critic_observations"]

            if self.playground_mode:
                # Extract and store only the privileged part
                privileged_observations = critic_observations[:, self.n_obs :]
                next_privileged_observations = next_critic_observations[:, self.n_obs :]
                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # Store full critic observations
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations

        self.ptr += 1

    def _gather_critic_observations(self, indices, observations, next_observations):
        """Helper method to gather critic observations based on mode."""
        if not self.asymmetric_obs:
            return None, None

        if self.playground_mode:
            # Gather privileged observations
            priv_obs_indices = indices.unsqueeze(-1).expand(
                -1, -1, self.privileged_obs_size
            )
            privileged_observations = torch.gather(
                self.privileged_observations, 1, priv_obs_indices
            ).reshape(self.n_env * indices.shape[1], self.privileged_obs_size)
            next_privileged_observations = torch.gather(
                self.next_privileged_observations, 1, priv_obs_indices
            ).reshape(self.n_env * indices.shape[1], self.privileged_obs_size)

            # Concatenate with regular observations to form full critic observations
            critic_observations = torch.cat(
                [observations, privileged_observations], dim=1
            )
            next_critic_observations = torch.cat(
                [next_observations, next_privileged_observations], dim=1
            )
        else:
            # Gather full critic observations
            critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
            critic_observations = torch.gather(
                self.critic_observations, 1, critic_obs_indices
            ).reshape(self.n_env * indices.shape[1], self.n_critic_obs)
            next_critic_observations = torch.gather(
                self.next_critic_observations, 1, critic_obs_indices
            ).reshape(self.n_env * indices.shape[1], self.n_critic_obs)

        return critic_observations, next_critic_observations

    def sample(self, batch_size: int):
        """Sample transitions from the buffer. To be implemented by subclasses."""
        raise NotImplementedError


class SimpleReplayBuffer(BaseReplayBuffer):
    """Simple replay buffer for single-step transitions."""

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        device=None,
    ):
        super().__init__(
            n_env=n_env,
            buffer_size=buffer_size,
            n_obs=n_obs,
            n_act=n_act,
            n_critic_obs=n_critic_obs,
            asymmetric_obs=asymmetric_obs,
            playground_mode=playground_mode,
            device=device,
        )

    @torch.no_grad()
    def sample(self, batch_size: int):
        """Sample single-step transitions from the buffer."""
        indices = torch.randint(
            0,
            min(self.buffer_size, self.ptr),
            (self.n_env, batch_size),
            device=self.device,
        )

        obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
        act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

        observations = torch.gather(self.observations, 1, obs_indices).reshape(
            self.n_env * batch_size, self.n_obs
        )
        next_observations = torch.gather(
            self.next_observations, 1, obs_indices
        ).reshape(self.n_env * batch_size, self.n_obs)
        actions = torch.gather(self.actions, 1, act_indices).reshape(
            self.n_env * batch_size, self.n_act
        )

        rewards = torch.gather(self.rewards, 1, indices).reshape(
            self.n_env * batch_size
        )
        dones = torch.gather(self.dones, 1, indices).reshape(self.n_env * batch_size)
        terminations = torch.gather(self.terminations, 1, indices).reshape(
            self.n_env * batch_size
        )
        time_outs = torch.gather(self.time_outs, 1, indices).reshape(
            self.n_env * batch_size
        )

        effective_n_steps = torch.ones_like(dones)

        # Handle asymmetric observations
        critic_observations, next_critic_observations = (
            self._gather_critic_observations(indices, observations, next_observations)
        )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "terminations": terminations,
                "time_outs": time_outs,
                "next_observations": next_observations,
                "effective_n_steps": effective_n_steps,
            },
            batch_size=self.n_env * batch_size,
        )

        # Validate outputs
        for k, v in out.items():
            if v.isnan().any():
                raise ValueError(f"{k} nan")
            if v.isinf().any():
                raise ValueError(f"{k} inf")

        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next_critic_observations"] = next_critic_observations

        return out


class NStepReplayBuffer(BaseReplayBuffer):
    """N-step replay buffer that supports n-step returns and discounting."""

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        super().__init__(
            n_env=n_env,
            buffer_size=buffer_size,
            n_obs=n_obs,
            n_act=n_act,
            n_critic_obs=n_critic_obs,
            asymmetric_obs=asymmetric_obs,
            playground_mode=playground_mode,
            device=device,
        )
        assert n_steps > 1, "NStepReplayBuffer requires n_steps > 1"
        self.gamma = gamma
        self.n_steps = n_steps

    @torch.no_grad()
    def sample(self, batch_size: int):
        """Sample n-step transitions from the buffer."""
        # Sample base indices
        if self.ptr >= self.buffer_size:
            # When the buffer is full, there is no protection against sampling across different episodes
            # We avoid this by temporarily setting self.pos - 1 to truncated = True if not done
            # https://github.com/DLR-RM/stable-baselines3/blob/b91050ca94f8bce7a0285c91f85da518d5a26223/stable_baselines3/common/buffers.py#L857-L860
            # TODO (Younggyo): Change the reference when this SB3 branch is merged
            current_pos = self.ptr % self.buffer_size
            curr_time_outs = self.time_outs[:, current_pos - 1].clone()
            self.time_outs[:, current_pos - 1] = torch.logical_not(
                self.dones[:, current_pos - 1]
            )
            indices = torch.randint(
                0,
                self.buffer_size,
                (self.n_env, batch_size),
                device=self.device,
            )
        else:
            # Buffer not full - ensure n-step sequence doesn't exceed valid data
            max_start_idx = max(1, self.ptr - self.n_steps + 1)
            indices = torch.randint(
                0,
                max_start_idx,
                (self.n_env, batch_size),
                device=self.device,
            )

        obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
        act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

        # Get base transitions
        observations = torch.gather(self.observations, 1, obs_indices).reshape(
            self.n_env * batch_size, self.n_obs
        )
        actions = torch.gather(self.actions, 1, act_indices).reshape(
            self.n_env * batch_size, self.n_act
        )

        # Handle asymmetric observations for base transitions
        critic_observations, _ = self._gather_critic_observations(
            indices, observations, None
        )

        # Create sequential indices for each sample
        # This creates a [n_env, batch_size, n_step] tensor of indices
        seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
        all_indices = (
            indices.unsqueeze(-1) + seq_offsets
        ) % self.buffer_size  # [n_env, batch_size, n_step]

        # Gather all rewards and terminal flags
        # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
        all_rewards = torch.gather(
            self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
        )
        all_dones = torch.gather(
            self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
        )
        all_time_outs = torch.gather(
            self.time_outs.unsqueeze(-1).expand(-1, -1, self.n_steps),
            1,
            all_indices,
        )

        # Create masks for rewards *after* first done
        # This creates a cumulative product that zeroes out rewards after the first done
        all_dones_shifted = torch.cat(
            [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
        )  # First reward should not be masked
        done_masks = torch.cumprod(
            1.0 - all_dones_shifted, dim=2
        )  # [n_env, batch_size, n_step]
        effective_n_steps = done_masks.sum(2)

        # Create discount factors
        discounts = torch.pow(
            self.gamma, torch.arange(self.n_steps, device=self.device)
        )  # [n_steps]

        # Apply masks and discounts to rewards
        masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
        discounted_rewards = masked_rewards * discounts.view(
            1, 1, -1
        )  # [n_env, batch_size, n_step]

        # Sum rewards along the n_step dimension
        n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

        # Find index of first done or truncation or last step for each sequence
        first_done = torch.argmax((all_dones > 0).float(), dim=2)  # [n_env, batch_size]
        first_time_out = torch.argmax(
            (all_time_outs > 0).float(), dim=2
        )  # [n_env, batch_size]

        # Handle case where there are no dones or truncations
        no_dones = all_dones.sum(dim=2) == 0
        no_time_outs = all_time_outs.sum(dim=2) == 0

        # When no dones or truncs, use the last index
        first_done = torch.where(no_dones, self.n_steps - 1, first_done)
        first_time_out = torch.where(no_time_outs, self.n_steps - 1, first_time_out)

        # Take the minimum (first) of done or truncation
        final_indices = torch.minimum(first_done, first_time_out)  # [n_env, batch_size]

        # Create indices to gather the final next observations
        final_next_obs_indices = torch.gather(
            all_indices, 2, final_indices.unsqueeze(-1)
        ).squeeze(-1)  # [n_env, batch_size]

        # Gather final values
        final_next_observations = self.next_observations.gather(
            1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
        )
        final_dones = self.dones.gather(1, final_next_obs_indices)
        final_terminations = self.terminations.gather(1, final_next_obs_indices)
        final_time_outs = self.time_outs.gather(1, final_next_obs_indices)

        # Handle asymmetric observations for final next observations
        next_critic_observations = None
        if self.asymmetric_obs:
            if self.playground_mode:
                # Gather final privileged observations
                final_next_privileged_observations = (
                    self.next_privileged_observations.gather(
                        1,
                        final_next_obs_indices.unsqueeze(-1).expand(
                            -1, -1, self.privileged_obs_size
                        ),
                    )
                )

                # Reshape for output
                next_privileged_observations = (
                    final_next_privileged_observations.reshape(
                        self.n_env * batch_size, self.privileged_obs_size
                    )
                )

                # Concatenate with next observations to form full next critic observations
                next_observations_reshaped = final_next_observations.reshape(
                    self.n_env * batch_size, self.n_obs
                )
                next_critic_observations = torch.cat(
                    [next_observations_reshaped, next_privileged_observations],
                    dim=1,
                )
            else:
                # Gather final next critic observations directly
                final_next_critic_observations = self.next_critic_observations.gather(
                    1,
                    final_next_obs_indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    ),
                )
                next_critic_observations = final_next_critic_observations.reshape(
                    self.n_env * batch_size, self.n_critic_obs
                )

        # Reshape everything to batch dimension
        rewards = n_step_rewards.reshape(self.n_env * batch_size)
        dones = final_dones.reshape(self.n_env * batch_size)
        terminations = final_terminations.reshape(self.n_env * batch_size)
        time_outs = final_time_outs.reshape(self.n_env * batch_size)
        effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
        next_observations = final_next_observations.reshape(
            self.n_env * batch_size, self.n_obs
        )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "terminations": terminations,
                "time_outs": time_outs,
                "next_observations": next_observations,
                "effective_n_steps": effective_n_steps,
            },
            batch_size=self.n_env * batch_size,
        )

        # Validate outputs
        for k, v in out.items():
            if v.isnan().any():
                raise ValueError(f"{k} nan")
            if v.isinf().any():
                raise ValueError(f"{k} inf")

        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next_critic_observations"] = next_critic_observations

        if self.n_steps > 1 and self.ptr >= self.buffer_size:
            # Roll back the truncation flags introduced for safe sampling
            self.time_outs[:, current_pos - 1] = curr_time_outs

        return out


# Keep the original SimpleReplayBuffer as SimpleReplayBufferOriginal for backward compatibility
class SimpleReplayBufferOriginal(nn.Module):
    """Original SimpleReplayBuffer implementation for backward compatibility.

    This class maintains the original interface that supported both single-step and n-step sampling.
    It's recommended to use SimpleReplayBuffer or NStepReplayBuffer instead.
    """

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        """
        A simple replay buffer that stores transitions in a circular buffer.
        Supports n-step returns and asymmetric observations.

        When playground_mode=True, critic_observations are treated as a concatenation of
        regular observations and privileged observations, and only the privileged part is stored
        to save memory.

        TODO (Younggyo): Refactor to split this into SimpleReplayBuffer and NStepReplayBuffer
        """
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size

        # Normalize observation shapes to always be tuples for consistent handling
        self.n_obs = n_obs if isinstance(n_obs, (tuple, list)) else (n_obs,)
        self.n_act = n_act
        self.n_critic_obs = (
            n_critic_obs if isinstance(n_critic_obs, (tuple, list)) else (n_critic_obs,)
        )

        # Store original shapes as integers/tuples for backward compatibility
        self.n_obs_original = n_obs
        self.n_critic_obs_original = n_critic_obs

        self.asymmetric_obs = asymmetric_obs
        self.playground_mode = playground_mode and asymmetric_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.observations = torch.zeros(
            (n_env, buffer_size, *self.n_obs), device=device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (n_env, buffer_size, n_act),
            device=device,
            dtype=torch.float,
        )
        self.rewards = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.terminations = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.time_outs = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.next_observations = torch.zeros(
            (n_env, buffer_size, *self.n_obs), device=device, dtype=torch.float
        )

        if asymmetric_obs:
            if self.playground_mode:
                # Only store the privileged part of observations (n_critic_obs - n_obs)
                # Handle both vector and image observations
                if isinstance(self.n_obs_original, int) and isinstance(
                    self.n_critic_obs_original, int
                ):
                    # Vector observations: simple subtraction
                    self.privileged_obs_size = (
                        self.n_critic_obs_original - self.n_obs_original
                    )
                    self.privileged_obs_shape = (self.privileged_obs_size,)
                elif len(self.n_obs) == 1 and len(self.n_critic_obs) == 1:
                    # Vector observations stored as tuples
                    self.privileged_obs_size = self.n_critic_obs[0] - self.n_obs[0]
                    self.privileged_obs_shape = (self.privileged_obs_size,)
                else:
                    # For image observations, assume critic obs includes extra features at the end
                    # This is a more complex case that might need custom handling
                    raise NotImplementedError(
                        "Playground mode with image observations requires specific implementation"
                    )

                self.privileged_observations = torch.zeros(
                    (n_env, buffer_size, *self.privileged_obs_shape),
                    device=device,
                    dtype=torch.float,
                )
                self.next_privileged_observations = torch.zeros(
                    (n_env, buffer_size, *self.privileged_obs_shape),
                    device=device,
                    dtype=torch.float,
                )
            else:
                # Store full critic observations
                self.critic_observations = torch.zeros(
                    (n_env, buffer_size, *self.n_critic_obs),
                    device=device,
                    dtype=torch.float,
                )
                self.next_critic_observations = torch.zeros(
                    (n_env, buffer_size, *self.n_critic_obs),
                    device=device,
                    dtype=torch.float,
                )
        self.ptr = 0

    @torch.no_grad()
    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["rewards"]
        dones = tensor_dict["dones"]
        terminations = tensor_dict["terminations"]
        time_outs = tensor_dict["time_outs"]
        next_observations = tensor_dict["next_observations"]

        for k, v in tensor_dict.items():
            if v.isnan().any():
                raise ValueError(f"{k} nan")
            if v.isinf().any():
                raise ValueError(f"{k} inf")
        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.terminations[:, ptr] = terminations
        self.time_outs[:, ptr] = time_outs
        self.next_observations[:, ptr] = next_observations

        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next_critic_observations"]

            if self.playground_mode:
                # Extract and store only the privileged part
                # Handle both vector and image observations
                if len(self.n_obs) == 1:
                    # Vector observations: slice from the end
                    privileged_observations = critic_observations[:, self.n_obs[0] :]
                    next_privileged_observations = next_critic_observations[
                        :, self.n_obs[0] :
                    ]
                else:
                    # For image observations, this would need custom logic
                    raise NotImplementedError(
                        "Playground mode privileged observation extraction for image obs not implemented"
                    )

                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # Store full critic observations
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1

    def _create_obs_indices(self, indices):
        """Helper method to create observation indices for both vector and image observations."""
        obs_indices = indices.unsqueeze(-1)
        # Add dimensions for each observation dimension
        for i in range(len(self.n_obs) - 1):
            obs_indices = obs_indices.unsqueeze(-1)
        return obs_indices.expand(-1, -1, *self.n_obs)

    def _create_critic_obs_indices(self, indices):
        """Helper method to create critic observation indices."""
        critic_obs_indices = indices.unsqueeze(-1)
        # Add dimensions for each critic observation dimension
        for i in range(len(self.n_critic_obs) - 1):
            critic_obs_indices = critic_obs_indices.unsqueeze(-1)
        return critic_obs_indices.expand(-1, -1, *self.n_critic_obs)

    @torch.no_grad()
    def sample(self, batch_size: int):
        # we will sample n_env * batch_size transitions

        if self.n_steps == 1:
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )

            obs_indices = self._create_obs_indices(indices)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, *self.n_obs
            )
            next_observations = torch.gather(
                self.next_observations, 1, obs_indices
            ).reshape(self.n_env * batch_size, *self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )

            rewards = torch.gather(self.rewards, 1, indices).reshape(
                self.n_env * batch_size
            )
            dones = torch.gather(self.dones, 1, indices).reshape(
                self.n_env * batch_size
            )
            terminations = torch.gather(self.terminations, 1, indices).reshape(
                self.n_env * batch_size
            )

            time_outs = torch.gather(self.time_outs, 1, indices).reshape(
                self.n_env * batch_size
            )

            effective_n_steps = torch.ones_like(dones)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, *self.privileged_obs_shape
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, *self.privileged_obs_shape)
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, *self.privileged_obs_shape)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                    next_critic_observations = torch.cat(
                        [next_observations, next_privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = self._create_critic_obs_indices(indices)
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, *self.n_critic_obs)
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, *self.n_critic_obs)
        else:
            # Sample base indices
            if self.ptr >= self.buffer_size:
                current_pos = self.ptr % self.buffer_size
                curr_time_outs = self.time_outs[:, current_pos - 1].clone()
                self.time_outs[:, current_pos - 1] = torch.logical_not(
                    self.dones[:, current_pos - 1]
                )
                indices = torch.randint(
                    0,
                    self.buffer_size,
                    (self.n_env, batch_size),
                    device=self.device,
                )
            else:
                # Buffer not full - ensure n-step sequence doesn't exceed valid data
                max_start_idx = max(1, self.ptr - self.n_steps + 1)
                indices = torch.randint(
                    0,
                    max_start_idx,
                    (self.n_env, batch_size),
                    device=self.device,
                )

            obs_indices = self._create_obs_indices(indices)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # Get base transitions
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, *self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, *self.privileged_obs_shape
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, *self.privileged_obs_shape)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = self._create_critic_obs_indices(indices)
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, *self.n_critic_obs)

            # Create sequential indices for each sample
            # This creates a [n_env, batch_size, n_step] tensor of indices
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (
                indices.unsqueeze(-1) + seq_offsets
            ) % self.buffer_size  # [n_env, batch_size, n_step]

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_time_outs = torch.gather(
                self.time_outs.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards *after* first done
            # This creates a cumulative product that zeroes out rewards after the first done
            all_dones_shifted = torch.cat(
                [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
            )  # First reward should not be masked
            done_masks = torch.cumprod(
                1.0 - all_dones_shifted, dim=2
            )  # [n_env, batch_size, n_step]
            effective_n_steps = done_masks.sum(2)

            # Create discount factors
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )  # [n_steps]

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * discounts.view(
                1, 1, -1
            )  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax(
                (all_dones > 0).float(), dim=2
            )  # [n_env, batch_size]
            first_time_out = torch.argmax(
                (all_time_outs > 0).float(), dim=2
            )  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_time_outs = all_time_outs.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_time_out = torch.where(no_time_outs, self.n_steps - 1, first_time_out)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(
                first_done, first_time_out
            )  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(-1)  # [n_env, batch_size]

            # Create properly shaped indices for next observations
            _final_next_obs_indices = final_next_obs_indices.unsqueeze(-1)
            for i in range(len(self.n_obs) - 1):
                _final_next_obs_indices = _final_next_obs_indices.unsqueeze(-1)

            # Gather final values
            final_next_observations = self.next_observations.gather(
                1, _final_next_obs_indices.expand(-1, -1, *self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_terminations = self.terminations.gather(1, final_next_obs_indices)
            final_time_outs = self.time_outs.gather(1, final_next_obs_indices)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather final privileged observations
                    final_next_privileged_observations = (
                        self.next_privileged_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, *self.privileged_obs_shape
                            ),
                        )
                    )

                    # Reshape for output
                    next_privileged_observations = (
                        final_next_privileged_observations.reshape(
                            self.n_env * batch_size, *self.privileged_obs_shape
                        )
                    )

                    # Concatenate with next observations to form full next critic observations
                    next_observations_reshaped = final_next_observations.reshape(
                        self.n_env * batch_size, *self.n_obs
                    )
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations],
                        dim=1,
                    )
                else:
                    # Gather final next critic observations directly
                    _final_next_critic_obs_indices = final_next_obs_indices.unsqueeze(
                        -1
                    )
                    for i in range(len(self.n_critic_obs) - 1):
                        _final_next_critic_obs_indices = (
                            _final_next_critic_obs_indices.unsqueeze(-1)
                        )

                    final_next_critic_observations = (
                        self.next_critic_observations.gather(
                            1,
                            _final_next_critic_obs_indices.expand(
                                -1, -1, *self.n_critic_obs
                            ),
                        )
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, *self.n_critic_obs
                    )

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            terminations = final_terminations.reshape(self.n_env * batch_size)
            time_outs = final_time_outs.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, *self.n_obs
            )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "terminations": terminations,
                "time_outs": time_outs,
                "next_observations": next_observations,
                "effective_n_steps": effective_n_steps,
            },
            batch_size=self.n_env * batch_size,
        )
        for k, v in out.items():
            if v.isnan().any():
                raise ValueError(f"{k} nan")
            if v.isinf().any():
                raise ValueError(f"{k} inf")
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next_critic_observations"] = next_critic_observations

        if self.n_steps > 1 and self.ptr >= self.buffer_size:
            # Roll back the truncation flags introduced for safe sampling
            self.time_outs[:, current_pos - 1] = curr_time_outs
        return out


if __name__ == "__main__":
    print("Testing SimpleReplayBufferOriginal with sequential values...")

    # Test 1: Vector observations
    print("\n=== Test 1: Vector Observations ===")
    buffer_vector = SimpleReplayBufferOriginal(
        n_env=4,  # Small number for easier tracking
        buffer_size=5,  # Small buffer size to see circular behavior
        n_obs=8,  # Vector observation
        n_act=2,
        n_steps=1,
        n_critic_obs=8,
        device="cpu",
    )

    # Add data with clear sequential patterns
    for step in range(7):  # More steps than buffer size to test circular buffer
        print(f"Adding step {step}")

        # Create observations where each env has a different base value
        # and each step adds to it
        observations = torch.zeros(buffer_vector.n_env, buffer_vector.n_obs[0])
        for env_idx in range(buffer_vector.n_env):
            # Each env gets: env_idx * 100 + step
            # So env 0 step 0 = 0, env 0 step 1 = 1, env 1 step 0 = 100, env 1 step 1 = 101, etc.
            observations[env_idx] = env_idx * 100 + step

        buffer_vector.extend(
            TensorDict(
                observations=observations,
                actions=torch.full(
                    (buffer_vector.n_env, buffer_vector.n_act), step
                ),  # Actions = step number
                rewards=torch.full(
                    (buffer_vector.n_env,), step
                ),  # Rewards = step number
                dones=torch.zeros(buffer_vector.n_env, dtype=torch.long),
                terminations=torch.zeros(buffer_vector.n_env, dtype=torch.long),
                time_outs=torch.zeros(buffer_vector.n_env, dtype=torch.long),
                next_observations=observations + 0.5,  # Next obs = obs + 0.5
                batch_size=(buffer_vector.n_env,),
            )
        )

    # Sample and show results
    print(f"\nBuffer pointer: {buffer_vector.ptr}")
    sample = buffer_vector.sample(batch_size=2)  # Sample 2 per env = 8 total samples

    print("Sampled observations (first few elements of each sample):")
    print("Sample keys:", list(sample.keys()))
    print("Sample batch size:", sample.batch_size)

    for i in range(min(8, sample["observations"].shape[0])):
        obs_first_3 = sample["observations"][i, :3]
        action = sample["actions"][i, 0]
        reward = sample["rewards"][i]
        print(
            f"Sample {i}: obs[0:3]={obs_first_3.tolist()}, action={action.item()}, reward={reward.item()}"
        )

    # Test 2: Image observations
    print("\n=== Test 2: Image Observations ===")
    buffer_image = SimpleReplayBufferOriginal(
        n_env=2,  # Even smaller for image test
        buffer_size=4,
        n_obs=(3, 4, 4),  # Small image: 3 channels, 4x4
        n_act=2,
        n_steps=1,
        n_critic_obs=(3, 4, 4),
        device="cpu",
    )

    for step in range(6):
        print(f"Adding image step {step}")

        # Create image observations where each pixel value = env_idx * 100 + step
        observations = torch.zeros(buffer_image.n_env, *buffer_image.n_obs)
        for env_idx in range(buffer_image.n_env):
            # Fill the entire image with the same value for easy identification
            observations[env_idx] = env_idx * 100 + step

        buffer_image.extend(
            TensorDict(
                observations=observations,
                actions=torch.full((buffer_image.n_env, buffer_image.n_act), step),
                rewards=torch.full((buffer_image.n_env,), step),
                dones=torch.zeros(buffer_image.n_env, dtype=torch.long),
                terminations=torch.zeros(buffer_image.n_env, dtype=torch.long),
                time_outs=torch.zeros(buffer_image.n_env, dtype=torch.long),
                next_observations=observations + 0.5,
                batch_size=(buffer_image.n_env,),
            )
        )

    # Sample and show results
    print(f"\nImage buffer pointer: {buffer_image.ptr}")
    sample_image = buffer_image.sample(batch_size=2)

    print("Sampled image observations (corner pixel values):")
    for i in range(min(4, sample_image["observations"].shape[0])):
        # Show the corner pixel value (all pixels should be the same)
        corner_pixel = sample_image["observations"][
            i, 0, 0, 0
        ]  # Channel 0, pixel (0,0)
        action = sample_image["actions"][i, 0]
        reward = sample_image["rewards"][i]
        print(
            f"Sample {i}: corner_pixel={corner_pixel.item()}, action={action.item()}, reward={reward.item()}"
        )

    # Test 3: N-step sampling
    print("\n=== Test 3: N-step Sampling ===")
    buffer_nstep = SimpleReplayBufferOriginal(
        n_env=2,
        buffer_size=10,
        n_obs=4,  # Vector observation
        n_act=1,
        n_steps=3,  # 3-step returns
        n_critic_obs=4,
        gamma=0.9,
        device="cpu",
    )

    for step in range(8):
        observations = torch.zeros(buffer_nstep.n_env, buffer_nstep.n_obs[0])
        for env_idx in range(buffer_nstep.n_env):
            observations[env_idx] = (
                env_idx * 1000 + step
            )  # Even bigger numbers for n-step

        buffer_nstep.extend(
            TensorDict(
                observations=observations,
                actions=torch.full((buffer_nstep.n_env, buffer_nstep.n_act), step),
                rewards=torch.full(
                    (buffer_nstep.n_env,), step + 1
                ),  # Reward = step + 1
                dones=torch.zeros(buffer_nstep.n_env, dtype=torch.long),
                terminations=torch.zeros(buffer_nstep.n_env, dtype=torch.long),
                time_outs=torch.zeros(buffer_nstep.n_env, dtype=torch.long),
                next_observations=observations + 100,  # Next obs = obs + 100
                batch_size=(buffer_nstep.n_env,),
            )
        )

    print(f"N-step buffer pointer: {buffer_nstep.ptr}")
    sample_nstep = buffer_nstep.sample(batch_size=2)

    print("N-step sampling results:")
    for i in range(min(4, sample_nstep["observations"].shape[0])):
        obs_val = sample_nstep["observations"][i, 0]
        next_obs_val = sample_nstep["next_observations"][i, 0]
        action = sample_nstep["actions"][i, 0]
        reward = sample_nstep["rewards"][i]
        n_steps = sample_nstep["effective_n_steps"][i]
        print(
            f"Sample {i}: obs={obs_val.item()}, next_obs={next_obs_val.item()}, "
            f"action={action.item()}, n_step_reward={reward.item():.2f}, effective_n_steps={n_steps.item()}"
        )

    print("\n=== All tests completed! ===")
