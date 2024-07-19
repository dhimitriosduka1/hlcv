# data_processing.py
from datasets import load_dataset
import torch
from torchvision import transforms
from PIL import Image

def create_transform(transform_config):
    """
    Create a torchvision transform pipeline based on the configuration.
    """
    transform_list = []
    for transform in transform_config:
        name = transform['name']
        params = transform.get('params', {})
        transform_class = getattr(transforms, name)
        transform_list.append(transform_class(**params))
    
    return transforms.Compose(transform_list)

def get_transforms(config):
    """
    Create train and base transforms based on the configuration.
    """
    train_transforms = create_transform(config['transforms']['train'])
    base_transforms = create_transform(config['transforms']['base'])
    return train_transforms, base_transforms

def transform_and_encode(example, processor, transforms):
    """
    Apply transforms to an example and encode it using the processor.
    """
    processed_images = [transforms(x.convert("RGB")) for x in example['img']]
    inputs = processor(processed_images, return_tensors='pt')
    inputs['label'] = example['label']

    return inputs

def load_and_prepare_dataset(data_config, model_config, processor):
    # Load dataset from Hugging Face datasets or local files
    if data_config['source'] == 'huggingface':
        dataset = load_dataset(data_config['name'], data_config.get('subset'))
    else:
        # Implement custom data loading logic for local files
        raise NotImplementedError("Local file loading not implemented")

    # Get transforms
    train_transforms, base_transforms = get_transforms(data_config)

    # Prepare datasets
    processed_datasets = {}
    for split in dataset.keys():
        is_train = split == 'train'
        transforms_to_apply = train_transforms if is_train else base_transforms
        
        processed_datasets[split] = dataset[split].with_transform(
            lambda example: transform_and_encode(example, processor, transforms_to_apply)
        )

    return processed_datasets

def get_dataset_splits(processed_datasets):
    train_dataset = processed_datasets['train']
    eval_dataset = processed_datasets['validation'] if 'validation' in processed_datasets else processed_datasets.get('test')
    test_dataset = processed_datasets.get('test')

    return train_dataset, eval_dataset, test_dataset