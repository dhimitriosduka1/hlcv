from torch.utils.data import DataLoader
from torch.utils.data import Subset

from copy import deepcopy


class BaseDataModule:
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, heldout_split, **loader_kwargs):
        self.dataset = dataset
        self.loader_kwargs = loader_kwargs
        self.n_samples = len(self.dataset)

        ## See if the dataset should be broken into two (e.g. as in a train and validation split)
        self.heldout_split = heldout_split
        self.heldout_set = None
        if self.heldout_split > 0.0:
            self.dataset, self.heldout_set = self._split_data(self.heldout_split)

        # Same thing for the heldout set, just don't shulffe (as we usually don't want shuffling for val set)
        self.heldout_kwargs = deepcopy(self.loader_kwargs)
        self.heldout_kwargs.update(dict(shuffle=False))

    def get_loader(self):
        print(f"Initialization DataLoader for {len(self.dataset)} samples with {self.loader_kwargs}")
        return DataLoader(dataset=self.dataset, **self.loader_kwargs)

    def get_heldout_loader(self):
        assert self.heldout_set is not None, \
            'There was no heldout split created! make sure you use heldout_split argument during initialization'
        print(f"Initialization heldout DataLoader {len(self.heldout_set)} samples with {self.heldout_kwargs}")
        return DataLoader(dataset=self.heldout_set, **self.heldout_kwargs)

    def _split_data(self, split):
        if split == 0.0:
            return self.dataset, None
        
        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)
        
        len_train = self.n_samples - len_valid
        train_mask = list(range(len_train))
        train_dataset = Subset(self.dataset, train_mask)
        eval_mask = list(range(len_train, len_train + len_valid))
        eval_dataset = Subset(self.dataset, eval_mask)

        self.n_samples = len(train_dataset)

        return train_dataset, eval_dataset
