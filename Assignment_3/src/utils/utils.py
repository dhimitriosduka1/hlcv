import json
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
import torch

from .vis_utils import visualize_grid


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