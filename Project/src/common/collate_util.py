import torch

def collate_fn(batch):
    collated = {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

    if "hand_features" in batch[0]:
        collated['hand_features'] = torch.stack([torch.tensor(x['hand_features']) for x in batch])

    return collated
