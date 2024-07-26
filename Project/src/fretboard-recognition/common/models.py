import os

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from transformers import PretrainedConfig, PreTrainedModel


class FasterRCNN(PreTrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.trainable_backbone_layers = config.trainable_backbone_layers
        self.model_dir = config.model_dir
        if self.model_dir:
            os.environ["TORCH_HOME"] = self.model_dir

        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            trainable_backbone_layers=self.trainable_backbone_layers,
        )

    def get_transforms(self):
        """Returns the transforms defined for the pre-trained weights of the model."""
        return FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()

    def forward(self, pixel_values=None, labels=None):
        if self.training:
            if labels is not None:
                # During training, labels is a list of dicts
                outputs = self.model(pixel_values, labels)
                loss = sum(loss for loss in outputs.values())

                return {
                    "loss": loss,
                    "cls_loss": outputs["loss_classifier"],
                    "box_loss": outputs["loss_box_reg"],
                }
            else:
                raise ValueError("Labels must be provided during training")
        else:
            if labels is not None:
                outputs = self.model(pixel_values, labels)
                # Ensure we return a dictionary with 'loss' key for compatibility with Trainer
                if isinstance(outputs, dict):
                    if "loss" not in outputs:
                        outputs["loss"] = torch.tensor(0.0)  # Add a dummy loss if not present
                else:
                    # If outputs is not a dict, wrap it in a dict
                    outputs = {"loss": torch.tensor(0.0), "logits": outputs}
            else:
                outputs = self.model(pixel_values)

            return outputs
