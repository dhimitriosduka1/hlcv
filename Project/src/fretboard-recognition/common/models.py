import os

import torch
from common.configs import ObjectDetectorConfig
from transformers import PreTrainedModel


class ObjectDetector(PreTrainedModel):
    def __init__(self, config: ObjectDetectorConfig) -> None:
        super().__init__(config)
        self.model_type = config.model_type
        self.num_labels = config.num_labels
        self.trainable_backbone_layers = config.trainable_backbone_layers
        self.model_dir = config.model_dir
        if self.model_dir:
            os.environ["TORCH_HOME"] = self.model_dir

        self.model = config.get_model()(
            weights=config.get_weights().COCO_V1,
            trainable_backbone_layers=self.trainable_backbone_layers,
        )
        self.__config = config

    def get_transforms(self):
        """Returns the transforms defined for the pre-trained weights of the model."""
        return self.__config.get_weights().COCO_V1.transforms()

    def forward(self, pixel_values=None, labels=None):
        if self.training:
            if labels is not None:
                # During training, labels is a list of dicts
                outputs = self.model(pixel_values, labels)
                loss = sum(loss for loss in outputs.values())

                return {
                    "loss": loss,
                    "cls_loss": outputs.get("loss_classifier", torch.tensor(0.0)),
                    "box_loss": outputs.get("loss_box_reg", torch.tensor(0.0))
                }
            else:
                raise ValueError("Labels must be provided during training")
        else:
            if labels is not None:
                # Run the model in training model during evaluation, to get loss calculation
                changed_state = False
                if not self.model.training:
                    self.model.train()
                    changed_state = True

                # Disable gradients mimicking eval mode
                with torch.no_grad():
                    outputs = self.model(pixel_values, labels)

                if changed_state:
                    self.model.eval()

                # Return the predictions (instead of the loss, because we are in eval mode)
                logits_outputs = self.model(pixel_values, labels)

                # Calculate the loss
                loss = sum(loss for loss in outputs.values() if isinstance(loss, torch.Tensor))

                # Return the loss and the logits
                return {
                    "loss": loss,
                    "cls_loss": outputs.get("loss_classifier", torch.tensor(0.0)),
                    "box_loss": outputs.get("loss_box_reg", torch.tensor(0.0)),
                    "logits": logits_outputs,
                }
            else:
                outputs = self.model(pixel_values)

            return outputs
