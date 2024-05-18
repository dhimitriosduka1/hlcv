import json
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
import torch
from json import dumps as json_dumps

from .vis_utils import visualize_grid


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_toy_data(num_inputs, input_size):
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


def rel_error(x, y):
    """returns relative error"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def show_net_weights(net):
    W1 = net.params["W1"]
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype("uint8"))
    plt.gca().axis("off")
    plt.show()


def pretty_dict(
    d: dict,
    inline: bool = True,
    include_brackets: bool = True,
    quote_keys: bool = False,
    indent_keys: bool = False,
    nums_as_pct: bool = False,
) -> str:
    """Pretty print a dictionary with optional parameters for better readability."""
    separators = (",", ": ")
    pretty_str = json_dumps(d, separators=separators, indent=4)
    pretty_str = pretty_str.replace('"', "") if not quote_keys else pretty_str
    pretty_str = pretty_str[1:-1] if not include_brackets else pretty_str
    all_lines = pretty_str.split("\n")
    nonempty_lines = [
        line.strip() for line in all_lines if line.strip() != ""
    ]  # Filter out empty lines
    pretty_string = ""
    for i, line in enumerate(nonempty_lines):
        if indent_keys:
            pretty_string += "\t"
        elif inline:
            pretty_string += " " if i > 0 else ""

        if not nums_as_pct:
            pretty_string += line
        else:
            # Convert numbers to percentages
            line_parts = line.split(": ")
            key, value = line_parts
            value = value.replace(",", "")
            value = f"{float(value) * 100:.2f}%"
            pretty_string += f"{key}: {value}"

        pretty_string += "\n" if i < len(nonempty_lines) - 1 else ""

    if inline:
        pretty_string = pretty_string.replace("\n", "")

    return pretty_string
