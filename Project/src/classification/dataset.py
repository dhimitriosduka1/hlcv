import os
import json
import torch

from PIL import Image
from torch.utils.data import Dataset

class ChordsDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        
        # Create a mapping from image_id to annotations
        self.image_to_annotations = {}
        for annotation in self.annotations:
            image_id = annotation['image_id']
            if image_id not in self.image_to_annotations:
                self.image_to_annotations[image_id] = []
            self.image_to_annotations[image_id].append(annotation)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels for this image
        image_id = img_info['id']
        label = self.image_to_labels.get(image_id, [])
        
        # Convert labels to tensor
        label = torch.as_tensor(label, dtype=torch.int64)

        print(image, label)

        return image, label