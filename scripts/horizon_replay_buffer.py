from typing import List, Tuple, Union

import torch
from skrl.memories.torch import Memory


class CustomMemory(Memory):
    def __init__(
        self,
        memory_size: int,
        num_envs: int = 1,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        """Custom memory

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: 1)
        :type num_envs: int, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        """
        super().__init__(memory_size, num_envs, device)

    def sample(
        self, names: Tuple[str], batch_size: int, mini_batches: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample a batch from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        # ================================
        # - sample a batch from memory.
        #   It is possible to generate only the sampling indexes and call self.sample_by_index(...)
        # ================================


class HorizonMemory(Memory):
    def __init__(
        self,
        memory_size: int,
        horizon_length: int,
        num_envs: int = 1,
        device: Union[str, torch.device] = "cuda:0",
        gamma: float = 0.99,
        n_step: int = 3,
    ) -> None:
        """Horizon-based replay memory supporting both Dreamer and TDMPC

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param horizon_length: Length of the horizon for sequence sampling
        :type horizon_length: int
        :param num_envs: Number of parallel environments (default: 1)
        :type num_envs: int, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        :param gamma: Discount factor for multi-step returns
        :type gamma: float, optional
        :param n_step: Number of steps for multi-step returns
        :type n_step: int, optional
        """
        super().__init__(memory_size, num_envs, device)
        self.horizon_length = horizon_length
        self.gamma = gamma
        self.n_step = n_step
        self.transition_buffer = []
        self.episode_starts = []  # Track start indices of episodes

    def add_transition(self, transition: dict) -> None:
        """Add a single transition to the memory

        :param transition: Dictionary containing transition data with keys:
                         'state', 'action', 'reward', 'next_state', 'done'
        :type transition: dict
        """
        # Track episode boundaries
        if len(self.transition_buffer) == 0 or self.transition_buffer[-1]["done"]:
            self.episode_starts.append(len(self))

        self.transition_buffer.append(transition)
        
        if len(self.transition_buffer) >= self.n_step:
            # Compute multi-step return
            reward, done = 0.0, False
            for i, trans in enumerate(self.transition_buffer):
                reward += (self.gamma**i) * trans["reward"]
                done = done or trans["done"]
                if done:
                    break

            # Create n-step transition with additional sequence info
            n_step_transition = {
                "state": self.transition_buffer[0]["state"],
                "action": self.transition_buffer[0]["action"],
                "reward": reward,
                "next_state": self.transition_buffer[-1]["next_state"],
                "done": done,
                "sequence_pos": len(self) % self.horizon_length,  # Position in sequence
            }
            self.append(n_step_transition)

            # Remove the oldest transition
            self.transition_buffer.pop(0)
            
            # Clean up old episode starts
            while self.episode_starts and self.episode_starts[0] < len(self) - self.memory_size:
                self.episode_starts.pop(0)

    def sample(
        self, names: Tuple[str], batch_size: int, mini_batches: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample a batch of sequences from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of sequences to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional

        :return: List of mini-batches, where each mini-batch contains a list of tensors
                 with shape (batch_size, horizon_length, data_dim)
        :rtype: list of torch.Tensor lists
        """
        total_samples = len(self)
        if total_samples < self.horizon_length:
            raise RuntimeError(f"Not enough samples in memory. Need at least {self.horizon_length} samples.")

        samples = []
        for _ in range(mini_batches):
            # Sample starting indices that don't cross episode boundaries
            valid_starts = []
            for i in range(total_samples - self.horizon_length + 1):
                # Check if sequence crosses episode boundary
                sequence_end = i + self.horizon_length
                crosses_boundary = False
                for episode_start in self.episode_starts:
                    if i < episode_start < sequence_end:
                        crosses_boundary = True
                        break
                if not crosses_boundary:
                    valid_starts.append(i)

            if not valid_starts:
                raise RuntimeError("No valid sequences found that don't cross episode boundaries")

            # Sample batch_size sequences
            start_indices = torch.randint(0, len(valid_starts), (batch_size,))
            sequence_starts = [valid_starts[i] for i in start_indices]

            # Gather sequences for each tensor name
            batch_tensors = []
            for name in names:
                sequences = []
                for start_idx in sequence_starts:
                    sequence = []
                    for i in range(self.horizon_length):
                        tensor = self.tensors[name][start_idx + i]
                        sequence.append(tensor)
                    sequences.append(torch.stack(sequence))
                batch_tensors.append(torch.stack(sequences))

            samples.append(batch_tensors)

        return samples

    def append(self, transition: dict) -> None:
        """Append a transition to the memory

        :param transition: Dictionary containing transition data
        :type transition: dict
        """
        for name, value in transition.items():
            if name not in self.tensors:
                raise ValueError(f"Tensor {name} not found in memory")
            self.tensors[name].append(value)

    def __len__(self) -> int:
        """Get the number of stored transitions

        :return: Number of stored transitions
        :rtype: int
        """
        return len(next(iter(self.tensors.values())))
