from transformers import PretrainedConfig


class FasterRCNNDefaultConfig(PretrainedConfig):
    model_type = "faster_rcnn"

    def __init__(self, model_dir=None, num_labels=2, trainable_backbone_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.model_dir = model_dir
        self.trainable_backbone_layers = trainable_backbone_layers
