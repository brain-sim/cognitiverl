"""
Clean and Simple Base Agent Class

Features:
- Checkpointing
- EMA with torch.lerp
- Device and precision management
- Abstract methods for get_action and get_value
- build_networks method
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn


class BaseAgent(nn.Module, ABC):
    """
    Clean base class for all agents.

    Features:
    - Checkpointing
    - EMA support
    - Device and precision management
    - Abstract methods for get_action and get_value
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self._ema_agent = None
        self._ema_decay = 0.999

    @abstractmethod
    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action from input."""
        pass

    @abstractmethod
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-value from input."""
        pass

    def build_networks(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: type[nn.Module] = nn.ELU,
        output_activation: Optional[nn.Module] = None,
    ) -> nn.Sequential:
        """
        Build a neural network with given specifications.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer (optional)

        Returns:
            nn.Sequential network
        """
        layers = []

        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layer_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            layers.extend([nn.Linear(layer_input_dim, hidden_dim), activation()])

        # Output layer
        final_input_dim = hidden_dims[-1] if hidden_dims else input_dim
        layers.append(nn.Linear(final_input_dim, output_dim))

        # Optional output activation
        if output_activation is not None:
            layers.append(output_activation())

        return nn.Sequential(*layers)

    def to_device(self, device: Union[str, torch.device]) -> "BaseAgent":
        """Move model to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        return self.to(self.device)

    def set_precision(self, dtype: torch.dtype) -> "BaseAgent":
        """Set the precision of the model."""
        self.dtype = dtype
        return self.to(dtype)

    def create_ema_agent(self, decay: float = 0.999) -> "BaseAgent":
        """Create EMA copy of the agent."""
        ema_agent = copy.deepcopy(self)
        ema_agent.eval()

        # Disable gradients for EMA agent
        for param in ema_agent.parameters():
            param.requires_grad = False

        # Store decay rate
        ema_agent._ema_decay = decay

        self._ema_agent = ema_agent
        return ema_agent

    def update_ema(self, decay: float | None = None):
        """Update EMA weights using torch.lerp."""
        if self._ema_agent is None:
            raise ValueError("EMA agent not created. Call create_ema_agent() first.")

        if decay is None:
            decay = getattr(self._ema_agent, "_ema_decay", 0.999)

        decay = float(decay) if decay is not None else 0.999

        with torch.no_grad():
            for ema_param, param in zip(
                self._ema_agent.parameters(), self.parameters()
            ):
                ema_param.data.lerp_(param.data, 1.0 - decay)

    @property
    def ema(self) -> Optional["BaseAgent"]:
        """Get the EMA agent."""
        return self._ema_agent

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: int = 0,
        **kwargs,
    ):
        """Save checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "step": step,
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if self._ema_agent is not None:
            checkpoint["ema_model_state_dict"] = self._ema_agent.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_ema: bool = False,
        strict: bool = True,
    ) -> dict:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load model weights
        if load_ema and "ema_model_state_dict" in checkpoint:
            state_dict = checkpoint["ema_model_state_dict"]
        else:
            state_dict = checkpoint["model_state_dict"]

        self.load_state_dict(state_dict, strict=strict)

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint
