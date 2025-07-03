import argparse
import os
import random
from dataclasses import asdict, fields

try:
    import jax
except ImportError:
    pass

import numpy as np
import torch
import yaml
from termcolor import colored


def seed_everything(
    envs,
    seed,
    use_torch=False,
    use_jax=False,
    torch_deterministic=False,
):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if isinstance(envs, list):
        for env in envs:
            env.seed(seed=seed)
    else:
        envs.seed(seed=seed)
    if use_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
    if use_jax:
        key = jax.random.PRNGKey(seed)
        return key


def set_high_precision():
    torch.set_float32_matmul_precision("high")


def _dataclass_to_argparse(parser, dataclass_type, prefix="", defaults=None):
    """
    Add arguments to parser from dataclass fields. Optionally use a prefix for nested dataclasses.
    """
    for f in fields(dataclass_type):
        arg_name = f"--{prefix}{f.name}"
        arg_type = f.type
        print(prefix, f.name, arg_name)
        default = getattr(defaults, f.name) if defaults is not None else f.default
        # Handle bools as store_true/store_false
        if arg_type is bool:
            if default is True:
                parser.add_argument(
                    arg_name, action="store_false", dest=f.name, help="(default: True)"
                )
            else:
                parser.add_argument(
                    arg_name, action="store_true", dest=f.name, help="(default: False)"
                )
        else:
            parser.add_argument(
                arg_name,
                type=arg_type,
                default=argparse.SUPPRESS,
                help=f"(default: {default})",
            )


def _update_dict(d, u):
    """Recursively update dict d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_dict(d[k], v)
        else:
            d[k] = v
    return d


def load_args(ArgsClass, yaml_path=None, cli_args=None):
    """
    Loads arguments from dataclass defaults, then YAML, then CLI (in that order).
    Args:
        ArgsClass: The dataclass type to instantiate.
        yaml_path: Path to YAML file (optional, can be None).
        cli_args: List of CLI args (optional, defaults to sys.argv[1:]).
    Returns:
        An instance of ArgsClass with merged arguments.
    """
    # 1. Start with dataclass defaults
    default_args = ArgsClass()
    merged = asdict(default_args)

    # 2. If YAML, update
    if yaml_path is not None and os.path.isfile(yaml_path):
        with open(yaml_path, "r") as f:
            yaml_args = yaml.safe_load(f)
        if yaml_args:
            merged.update(yaml_args)

    # 3. CLI overrides
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file."
    )
    _dataclass_to_argparse(parser, ArgsClass, defaults=default_args)
    args_ns, unknown = parser.parse_known_args(cli_args)
    cli_dict = vars(args_ns)
    config_from_cli = cli_dict.pop("config", None)
    # If --config is given on CLI, reload YAML and update again
    if config_from_cli and (yaml_path is None or config_from_cli != yaml_path):
        if os.path.isfile(config_from_cli):
            with open(config_from_cli, "r") as f:
                yaml_args2 = yaml.safe_load(f)
            if yaml_args2:
                merged.update(yaml_args2)
    merged.update(cli_dict)
    # Return as ArgsClass instance
    return ArgsClass(**merged)


def update_learning_rate_adaptive(
    optimizer, kl_divergence, desired_kl, lr_multiplier, min_lr=1e-6, max_lr=1e-2
):
    current_lr = optimizer.param_groups[0]["lr"]

    if kl_divergence > desired_kl * 2.0:
        new_lr = current_lr / lr_multiplier
    elif kl_divergence < desired_kl / 2.0:
        new_lr = current_lr * lr_multiplier
    else:
        new_lr = current_lr

    # Clamp learning rate to reasonable bounds
    new_lr = np.clip(new_lr, min_lr, max_lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

    return new_lr


def print_dict(
    val,
    nesting: int = -4,
    start: bool = True,
    color: str = None,
    attrs: list[str] = None,
):
    """Outputs a nested dictionary."""
    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(colored(k, color, attrs=attrs), end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        # deal with functions in print statements
        if callable(val):
            print(colored(callable_to_string(val), color, attrs=attrs))
        else:
            print(colored(val, color, attrs=attrs))
