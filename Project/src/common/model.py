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
            label2id={c: str(i) for i, c in enumerate(labels)},
            output_hidden_states=True
        )
    else:
        model = AutoModelForImageClassification.from_pretrained(
            model_config['name'],
            num_labels=len(labels),
            ignore_mismatched_sizes=True,
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
            output_hidden_states=True
        )
    return model

# Source: https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def load_processor(model_config):
    return AutoImageProcessor.from_pretrained(model_config['name'])