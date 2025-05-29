import os
import random

import jax
import numpy as np
import torch


def seed_everything(
    envs, seed, use_torch=False, use_jax=False, torch_deterministic=False
):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
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
