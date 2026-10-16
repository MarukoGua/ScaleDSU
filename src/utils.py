import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    Args:
        seed (`int`): The seed to set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_rank(ranks, file_path, file_name):
    avg_rank = np.mean(ranks, axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(avg_rank)
    plt.legend()
    plt.savefig(os.path.join(file_path, file_name))
    plt.close()


def str2bool(v):
    if isinstance(v, str):
        return v.lower() in ['true', 'yes', '1']
    else:
        return v