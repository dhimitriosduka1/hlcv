from transformers import AutoImageProcessor, AutoModelForImageClassification

def load_model_and_processor(model_config):
    model = AutoModelForImageClassification.from_pretrained(
        model_config['name'],
        num_labels=model_config['num_labels'],
        ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained(model_config['name'])
    return model, processor