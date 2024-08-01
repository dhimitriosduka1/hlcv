from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    FCOS_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
)
from transformers import PretrainedConfig


class ObjectDetectorConfig(PretrainedConfig):
    AVAILABLE_MODEL_TYPES = [
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet50_fpn_v2",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "fcos_resnet50_fpn",
        "ssd300_vgg16",
        "retinanet_resnet50_fpn",
        "retinanet_resnet50_fpn_v2",
        "ssdlite320_mobilenet_v3_large",
    ]

    def __init__(
        self,
        name="fasterrcnn_resnet50_fpn",
        model_dir=None,
        num_labels=2,
        trainable_backbone_layers=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if name not in self.AVAILABLE_MODEL_TYPES:
            raise ValueError(
                f"Model type '{name}' not available. Choose one of {self.AVAILABLE_MODEL_TYPES}"
            )

        self.model_type = name
        self.num_labels = num_labels
        self.model_dir = model_dir
        self.trainable_backbone_layers = trainable_backbone_layers

    def get_model(self) -> nn.Module:
        """Returns the model class based on the model type, ready to instantiate."""
        if self.model_type == "fasterrcnn_resnet50_fpn":
            return fasterrcnn_resnet50_fpn
        elif self.model_type == "fasterrcnn_resnet50_fpn_v2":
            return fasterrcnn_resnet50_fpn_v2
        elif self.model_type == "fasterrcnn_mobilenet_v3_large_fpn":
            return fasterrcnn_mobilenet_v3_large_fpn
        elif self.model_type == "fasterrcnn_mobilenet_v3_large_320_fpn":
            return fasterrcnn_mobilenet_v3_large_320_fpn
        elif self.model_type == "fcos_resnet50_fpn":
            return fcos_resnet50_fpn
        elif self.model_type == "ssd300_vgg16":
            return ssd300_vgg16
        elif self.model_type == "retinanet_resnet50_fpn":
            return retinanet_resnet50_fpn
        elif self.model_type == "retinanet_resnet50_fpn_v2":
            return retinanet_resnet50_fpn_v2
        elif self.model_type == "ssdlite320_mobilenet_v3_large":
            return ssdlite320_mobilenet_v3_large
        else:
            raise ValueError(
                f"Model type '{self.model_type}' not available. Choose one of {self.AVAILABLE_MODEL_TYPES}"
            )

    def get_weights(self):
        """Returns the weights class based on the model type."""
        if self.model_type == "fasterrcnn_resnet50_fpn":
            return FasterRCNN_ResNet50_FPN_Weights
        elif self.model_type == "fasterrcnn_resnet50_fpn_v2":
            return FasterRCNN_ResNet50_FPN_V2_Weights
        elif self.model_type == "fasterrcnn_mobilenet_v3_large_fpn":
            return FasterRCNN_MobileNet_V3_Large_FPN_Weights
        elif self.model_type == "fasterrcnn_mobilenet_v3_large_320_fpn":
            return FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        elif self.model_type == "fcos_resnet50_fpn":
            return FCOS_ResNet50_FPN_Weights
        elif self.model_type == "ssd300_vgg16":
            return SSD300_VGG16_Weights
        elif self.model_type == "retinanet_resnet50_fpn":
            return RetinaNet_ResNet50_FPN_Weights
        elif self.model_type == "retinanet_resnet50_fpn_v2":
            return RetinaNet_ResNet50_FPN_V2_Weights
        elif self.model_type == "ssdlite320_mobilenet_v3_large":
            return SSDLite320_MobileNet_V3_Large_Weights
        else:
            raise ValueError(
                f"Model type '{self.model_type}' not available. Choose one of {self.AVAILABLE_MODEL_TYPES}"
            )
