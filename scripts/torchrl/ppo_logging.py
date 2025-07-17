"""Logging utilities for TorchRL PPO training following RSL-RL's approach."""

import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import yaml
from isaaclab.utils.io import dump_pickle, dump_yaml


def setup_log_dir(
    experiment_name: str,
    task_name: str,
    seed: int,
    run_name: Optional[str] = None,
    log_root: str = "logs/torchrl",
) -> str:
    """Set up logging directory structure following RSL-RL convention.
    
    Args:
        experiment_name: Name of the experiment (e.g., "ppo_continuous_action")
        task_name: Name of the task being trained
        seed: Random seed used
        run_name: Optional additional run name
        log_root: Root directory for logs
        
    Returns:
        Path to the log directory
    """
    # Create base log path: logs/torchrl/{experiment_name}
    log_root_path = os.path.join(log_root, experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # Create timestamped directory: {timestamp}_{task}_{seed}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir += f"_{task_name}_seed{seed}"
    
    if run_name:
        log_dir += f"_{run_name}"
        
    log_dir = os.path.join(log_root_path, log_dir)
    
    # Create subdirectories
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "videos", "train"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    
    return log_dir


def save_configs(log_dir: str, args: Any, env_cfg: Optional[Any] = None):
    """Save configuration files in both YAML and pickle formats.
    
    Args:
        log_dir: Directory to save configs
        args: Training arguments/configuration
        env_cfg: Optional environment configuration
    """
    params_dir = os.path.join(log_dir, "params")
    
    # Convert dataclass to dict if needed
    if hasattr(args, "__dict__") and not isinstance(args, dict):
        args_dict = vars(args)
    else:
        args_dict = args
        
    # Save args
    dump_yaml(os.path.join(params_dir, "args.yaml"), args_dict)
    dump_pickle(os.path.join(params_dir, "args.pkl"), args)
    
    # Save env config if provided
    if env_cfg is not None:
        dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
        dump_pickle(os.path.join(params_dir, "env.pkl"), env_cfg)


def get_video_kwargs(log_dir: str, video_interval: int, video_length: int) -> dict:
    """Get video recording kwargs for gym.wrappers.RecordVideo.
    
    Args:
        log_dir: Base log directory
        video_interval: Steps between video recordings
        video_length: Length of each video in steps
        
    Returns:
        Dictionary of kwargs for RecordVideo wrapper
    """
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "train"),
        "step_trigger": lambda step: step % video_interval == 0,
        "video_length": video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print(f"    video_folder: {video_kwargs['video_folder']}")
    print(f"    video_interval: {video_interval}")
    print(f"    video_length: {video_length}")
    return video_kwargs


def save_checkpoint(
    log_dir: str,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    iteration: int,
    args: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    ema_agent: Optional[torch.nn.Module] = None,
) -> str:
    """Save model checkpoint following RSL-RL convention.
    
    Args:
        log_dir: Base log directory
        agent: The agent model
        optimizer: The optimizer
        global_step: Current global step
        iteration: Current iteration
        args: Training arguments
        metrics: Optional metrics to save
        ema_agent: Optional EMA agent model
        
    Returns:
        Path to saved checkpoint
    """
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"model_{iteration}.pt")
    
    checkpoint_data = {
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "iteration": iteration,
        "args": args,
    }
    
    if metrics:
        checkpoint_data["metrics"] = metrics
        
    if ema_agent is not None:
        checkpoint_data["ema_model_state_dict"] = ema_agent.state_dict()
        
    torch.save(checkpoint_data, ckpt_path)
    
    # Also save as 'last_model.pt' for easy resuming
    last_path = os.path.join(ckpt_dir, "last_model.pt")
    torch.save(checkpoint_data, last_path)
    
    return ckpt_path


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load a checkpoint from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    return torch.load(checkpoint_path)


def integrate_logging_with_ppo(args, envs=None):
    """Helper function to integrate logging with existing PPO training.
    
    This modifies the args object to add log_dir and returns video_kwargs if needed.
    
    Args:
        args: Training arguments object
        envs: Optional environment for extracting config
        
    Returns:
        Tuple of (log_dir, video_kwargs or None)
    """
    # Set up log directory
    log_dir = setup_log_dir(
        experiment_name=args.exp_name,
        task_name=args.task,
        seed=args.seed,
        run_name=getattr(args, "run_name", None),
    )
    
    # Save configurations
    env_cfg = envs.unwrapped.cfg if envs and hasattr(envs.unwrapped, "cfg") else None
    save_configs(log_dir, args, env_cfg)
    
    # Get video kwargs if video recording is enabled
    video_kwargs = None
    if args.video:
        video_kwargs = get_video_kwargs(
            log_dir, 
            args.video_interval,
            args.video_length
        )
    
    return log_dir, video_kwargs