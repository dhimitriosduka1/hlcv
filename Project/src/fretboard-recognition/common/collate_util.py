
import torch


def collate_fn(batch: list) -> dict:
    """Collate function for the dataset. It takes a list of samples and returns a dictionary
    with the pixel values and the labels."""
    return {
        "pixel_values": torch.stack([item[0] for item in batch]),
        "labels": [item[1] for item in batch],
    }