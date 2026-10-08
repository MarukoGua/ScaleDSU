import os
import json
import random
import logging

import numpy as np
import torch
from prettytable import PrettyTable
import matplotlib.pyplot as plt

import data_util


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

def load_data(args, mode):
    train_dataset, valid_dataset, test_dataset = None, None, None
    load_func = data_util.DATA_TYPE[args.data]
    
    if mode=='train':
        logging.info('loading training data...')
        train_dataset, valid_dataset = load_func(args, mode=mode)
    
    
    if mode=='test':
        logging.info('loading test data...')
        test_dataset = load_func(args, mode=mode, num_trace=args.num_trace)
    
    return train_dataset, valid_dataset, test_dataset


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
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



def get_train_transform(gaussian_sigma=-1, shift_n=-1, scale_ratio=-1):
    transforms = []
    if gaussian_sigma != -1:
        transform = data_util.preprocess.add_gaussian_noise
        transforms.append(transform)
    
    if scale_ratio != -1:
        transforms.append(data_util.preprocess.scale)
    
    if shift_n != -1:
        transforms.append(data_util.preprocess.shift)

    return MultipleApply(transforms)
    
    
    
class MultipleApply:
    """Apply a list of transformations to an image and get multiple transformed images.

    Args:
        transforms (list or tuple): list of transformations

    Example:
        
        >>> transform1 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224)
        ... ])
        >>> transform2 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224),
        ... ])
        >>> multiply_transform = MultipleApply([transform1, transform2])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, kwargs):
        for t in self.transforms:
            x = t(x, **kwargs)
        return x
        # return [t(x) for t in self.transforms]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string