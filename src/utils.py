import os
import json
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
    # ^^ safe to call this function even if cuda is not available


def plot_rank(ranks, file_path, file_name):
    # x = [int(ranks[i][0]) for i in range(0, len(ranks))]
    # y = [int(ranks[i][1]) for i in range(0, len(ranks))]

    
    trace_needed = np.where(np.array(ranks)<=0)[0]
    if len(trace_needed) > 0:
        trace_needed = trace_needed[0]
    else:
        trace_needed = len(ranks)
    
    plt.figure(figsize=(10, 5))
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(ranks, label=f"Trace Needed: {trace_needed}")
    plt.legend()
    plt.savefig(os.path.join(file_path, file_name))
    plt.close()


def save_ids_to_json(dataset, json_file):
    if not dataset:
        return
    ids = [int(sample['id']) for sample in dataset]

    with open(json_file, "w") as f:
        json.dump(ids, f)


def str2bool(v):
    if isinstance(v, str):
        return v.lower() in ['true', 'yes', '1']
    else:
        return v


def count_parameters(model):
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        # table.add_row([name, params])
        total_params += params
    # logging.info(table)
    # logging.info(f"Total Trainable Params: {total_params}")
    return total_params
