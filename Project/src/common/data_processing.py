# data_processing.py
from datasets import load_dataset
from torchvision import transforms

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
    if 'transforms' not in config:
        return None, None

    train_transforms = create_transform(config['transforms']['train'])
    base_transforms = create_transform(config['transforms']['base'])
    return train_transforms, base_transforms

def transform_and_encode(batch, processor, transforms):
    """
    Apply transforms to an example and encode it using the processor.
    """
    if transforms != None:
        processed_images = [transforms(x.convert("RGB")) for x in batch['image']]
    else:
        processed_images = [x.convert("RGB") for x in batch['image']]

    inputs = processor(processed_images, return_tensors='pt')
    inputs['label'] = batch['label']

    return inputs

def load_and_prepare_dataset(data_config, processor):
    # Load dataset from Hugging Face datasets or local files
    if data_config['source'] == 'huggingface':
        dataset = load_dataset(data_config['name'])
    else:
        dataset = load_dataset(data_config['source'], data_dir=data_config['name'])

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
    
    extra_train_datasets = {}
    for dataset_name, dataset_info in data_config.get('additional_test_datasets', {}).items():
        if dataset_info['source'] == 'huggingface':
            dataset = load_dataset(dataset_info['name'])
        else:
            dataset = load_dataset(dataset_info['source'], data_dir=dataset_info['name'])

        # Only process the test split for additional datasets
        if 'test' in dataset:
            extra_train_datasets[dataset_name] = dataset['test'].with_transform(
                lambda example: transform_and_encode(example, processor, base_transforms)
            )

    return processed_datasets, extra_train_datasets

def get_dataset_splits(processed_datasets):
    train_dataset = processed_datasets['train']
    eval_dataset = processed_datasets['validation']
    test_dataset = processed_datasets['test']

    return train_dataset, eval_dataset, test_dataset
