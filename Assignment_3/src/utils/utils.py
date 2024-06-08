import json
import os
import random
from collections import OrderedDict
from gc import collect as garbage_collect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda import empty_cache as cuda_empty_cache
from torch.cuda import mem_get_info

from .vis_utils import visualize_grid


def clean(do_gpu: bool = True):
    garbage_collect() # release memory
    if do_gpu:
        cuda_empty_cache()
        mem_info = mem_get_info()
        print(
            f"Freeing GPU Memory\nFree: %d MB\tTotal: %d MB"
            % (mem_info[0] // 1024**2, mem_info[1] // 1024**2)
        )

def seed_everything(seed=0):    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))

    if n_gpu_use > 0:
        clean()
    
    return device, list_ids


class MetricTracker:
    """
    Keeps track of metrics by keeping the sum and the count, and returning the average.
    Also writes to the Writer at each update, if writer is given.
    """
    def __init__(self, keys=None, writer=None):
        self.writer = writer
        self.metrics_dict = {key: dict(count=0, sum=0.0) for key in keys}
        self.reset()

    def reset(self, keys=None):
        if keys is None:
            self.metrics_dict = {key: dict(count=0, sum=0.0) for key in self.metrics_dict}
        else:# Use new keys if given
            self.metrics_dict = {key: dict(count=0, sum=0.0) for key in keys}

    def update(self, key, value):
        assert key in self.metrics_dict, f"Key {key} wasn't given at initializaiton of the tracker"
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.metrics_dict[key]['count'] += 1
        self.metrics_dict[key]['sum'] += value

    def avg(self, key):
        return self.metrics_dict[key]['sum'] / self.metrics_dict[key]['count']

    def result(self):
        return {key: self.avg(key) for key in self.metrics_dict}
