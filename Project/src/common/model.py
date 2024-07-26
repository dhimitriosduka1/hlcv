import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTForImageClassification

def load_model(model_config, labels):
    if 'mae' in model_config['name']:
        model = ViTForImageClassification.from_pretrained(
            model_config['name'], 
            num_labels=len(labels),
            attn_implementation="sdpa", 
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True,
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    else:
        model = AutoModelForImageClassification.from_pretrained(
            model_config['name'],
            num_labels=len(labels),
            ignore_mismatched_sizes=True,
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
    return model

def load_processor(model_config):
    return AutoImageProcessor.from_pretrained(model_config['name'])