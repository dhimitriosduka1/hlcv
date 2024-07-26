import json
import os

from scipy import datasets
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_convert


class GuitarNeckDataset(Dataset):
    def __init__(self, root_dir, split="train", class_label=1, transforms=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(self.root_dir, "images"))))
        self.annotations = json.load(
            open(os.path.join(self.root_dir, "labels", "_annotations.coco.json"))
        )
        self.class_label = class_label

    @classmethod
    def from_dir(cls, root_dir, splits=["train", "valid", "test"], class_label=1, transforms=None):
        datasets = []
        for split in splits:
            datasets.append(cls(root_dir, split, class_label, transforms))
        return tuple(datasets)

    def __lookup_image_id(self, idx: int) -> int:
        """Lookup the `image_id` for a given index of an image."""
        img_name = self.imgs[idx]
        for image in self.annotations["images"]:
            if image["file_name"] == img_name:
                return image["id"]
        return None

    def __lookup_annotation_idx(self, image_id: int) -> int | None:
        """Lookup the indexes of annotations for a given `image_id`."""
        idxs = []
        for i, annotation in enumerate(self.annotations["annotations"]):
            if annotation["image_id"] == image_id:
                idxs.append(i)
        return idxs

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        annotation_idxs = self.__lookup_annotation_idx(self.__lookup_image_id(idx))

        bboxes = []
        for i in annotation_idxs:
            annotation = self.annotations["annotations"][i]
            bbox = annotation["bbox"]
            bboxes.append(bbox)

        # Ensure bbox is in [x_min, y_min, x_max, y_max] format
        bboxes = box_convert(torch.tensor(bboxes, dtype=torch.float32), "xywh", "xyxy")

        target = {}
        target["boxes"] = bboxes

        # Assuming only one class (guitar neck)
        target["labels"] = torch.tensor([self.class_label] * len(bboxes), dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
