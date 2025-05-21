import os
import random

import numpy as np
import torch


def seed_everything(envs, seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    envs.seed(seed=seed)


def set_high_precision():
    torch.set_float32_matmul_precision("high")
