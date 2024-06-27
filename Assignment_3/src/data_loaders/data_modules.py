import os
from torchvision import datasets

from .base_data_modules import BaseDataModule
from src.utils.transform_presets import presets


class CIFAR10DataModule(BaseDataModule):
    """
    CIFAR10 data loading using BaseDataModule
    """
    def __init__(self, data_dir, transform_preset, heldout_split=0.0, training=True, root_dir=None, **loader_kwargs):
        # Figure out the Transformation
        data_split = 'train' if training else 'eval'
        transform = presets[transform_preset][data_split]
        print(f"transforms for preset {transform_preset} for split {data_split} are {transform}")

        if root_dir is not None:
            data_dir = os.path.join(root_dir, data_dir)

        # Create the dataset!
        dataset = datasets.CIFAR10(
            root=data_dir, 
            train=training, 
            download=True, 
            transform=transform
        )

        # This will take care of splitting the data. It will also have get_loader() and get_heldout_loader()
        # implemented.
        super().__init__(dataset, heldout_split=heldout_split, **loader_kwargs)