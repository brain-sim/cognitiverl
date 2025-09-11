from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from tensordict import TensorDict

__all__ = [
    "make_image_grid",
    "process_image_batch",
    "log_tensor_shapes",
    "get_grad_norm",
    "print_section_header",
    "format_number",
    "compute_metrics",
    "compute_random_baseline",
]

# ============================================================================
# Utility Functions (simplified with TensorDict)
# ============================================================================


def compute_metrics(
    pred_actions: torch.Tensor, target_actions: torch.Tensor
) -> Dict[str, float]:
    """Compute MSE and cosine similarity between predictions and targets."""
    mse = F.mse_loss(pred_actions, target_actions).item()

    # Cosine similarity (higher is better)
    pred_flat = pred_actions.view(pred_actions.size(0), -1)
    target_flat = target_actions.view(target_actions.size(0), -1)
    cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean().item()

    return {"mse": mse, "cosine_similarity": cosine_sim}


def compute_random_baseline(target_actions: torch.Tensor) -> Dict[str, float]:
    """Compute metrics against random baseline."""
    random_actions = torch.randn_like(target_actions)
    return compute_metrics(random_actions, target_actions)


def print_section_header(title: str, char: str = "=", width: int = 80):
    """Print a nice section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def format_number(num):
    """Format large numbers with K/M suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def log_tensor_shapes(
    tensor_dict: TensorDict, prefix: str = ""
) -> Dict[str, List[int]]:
    """Log tensor shapes from TensorDict for debugging."""
    shapes = {}
    for key, tensor in tensor_dict.items():
        if tensor is not None:
            shapes[f"{prefix}{key}_shape"] = list(tensor.shape)
    return shapes


def get_grad_norm(model) -> float:
    """Get gradient norm for monitoring."""
    try:
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1.0 / 2)
    except Exception as e:
        print(f"Error getting gradient norm: {e}")
        return 0.0


def make_image_grid(
    images: torch.Tensor, nrow: int = 4, max_images: int = 8
) -> torch.Tensor:
    """Create image grid for logging (handles different formats).

    For sequence data: creates grid with samples as rows and timesteps as columns.
    """
    if images.dim() == 5:  # (B, T, C, H, W) - sequence format
        B, T, C, H, W = images.shape
        num_samples = min(max_images, B)

        # Select samples and reshape for grid: (num_samples * T, C, H, W)
        selected_images = images[:num_samples]  # (num_samples, T, C, H, W)

        # Reshape to show samples as rows, timesteps as columns
        grid_images = selected_images.view(num_samples * T, C, H, W)

        # Create grid with T columns (timesteps) and num_samples rows
        grid = torchvision.utils.make_grid(
            grid_images,
            nrow=T,  # T timesteps per row
            normalize=True,
            value_range=(0, 1) if grid_images.max() <= 1.0 else None,
        )
        return grid

    elif images.dim() == 4:  # (B, C, H, W) - single timestep
        if images.size(1) > 8:  # Likely (B, T*C, H, W) - need to reshape
            B, TxC, H, W = images.shape
            T = TxC // 3  # Assume 3 channels
            images = images.view(B, T, 3, H, W)
            # Now treat as sequence format
            return make_image_grid(images, nrow=nrow, max_images=max_images)
        else:
            # Regular (B, C, H, W) format
            images = images[:max_images]

    # Use shared image processing function for normalization
    images = process_image_batch(images, target_format="BCHW", normalize_to_01=True)

    return torchvision.utils.make_grid(images, nrow=nrow, normalize=True)


def process_image_batch(
    images: torch.Tensor,
    target_format: str = "BTCHW",
    normalize_to_01: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """Process image batch with format conversion and normalization.

    Args:
        images: Input image tensor
        target_format: Target format ("BTCHW" for 5D, "BCHW" for 4D)
        normalize_to_01: Whether to normalize from [0,255] to [0,1]
        device: Target device (optional)

    Returns:
        Processed image tensor
    """
    # Convert to float first
    if not images.dtype.is_floating_point:
        images = images.float()
    elif images.dtype != torch.float32:
        images = images.float()

    # Handle format conversion
    if target_format == "BTCHW" and images.dim() == 5:
        # Check if last dimension looks like channels
        if images.shape[-1] in (1, 3, 4):  # B,T,H,W,C → B,T,C,H,W
            images = images.permute(0, 1, 4, 2, 3)
    elif target_format == "BCHW" and images.dim() == 4:
        # Check if last dimension looks like channels
        if images.shape[-1] in (1, 3, 4):  # B,H,W,C → B,C,H,W
            images = images.permute(0, 3, 1, 2)

    # Normalize to [0,1] if needed
    if normalize_to_01 and images.max() > 1.0:
        images = (images / 255.0).clamp(0.0, 1.0)

    # Move to device if specified
    if device is not None:
        images = images.to(device, non_blocking=True)

    return images
