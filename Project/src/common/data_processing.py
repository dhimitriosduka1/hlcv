# data_processing.py
import mediapipe as mp
import numpy as np

from datasets import load_dataset
from torchvision import transforms

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_hand_features(image):
    results = hands.process(image)
    if results.multi_hand_landmarks:
        # Extract the 21 hand landmarks
        landmarks = results.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    else:
        return np.zeros(21 * 3)

def process_example(example):
    # Process image for vision model
    inputs = processor(images=example["image"], return_tensors="pt")
    
    # Extract hand features
    hand_features = extract_hand_features(example["image"])
    
    # Combine inputs
    inputs['hand_features'] = torch.tensor(hand_features)
    
    return inputs

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

def transform_and_encode(batch, processor, transforms, include_hand_features=False):
    """
    Apply transforms to an example and encode it using the processor.
    """
    if transforms != None:
        processed_images = [transforms(x.convert("RGB")) for x in batch['image']]
    else:
        processed_images = [x.convert("RGB") for x in batch['image']]

    inputs = processor(processed_images, return_tensors='pt')
    inputs['label'] = batch['label']
    
    if include_hand_features:
        # Extract hand features
        hand_poses = [extract_hand_features(np.array(x)) for x in batch['image']]
        inputs['hand_features'] = hand_poses

    return inputs

def load_and_prepare_dataset(data_config, processor, include_hand_features=False):
    # Load dataset from Hugging Face datasets or local files
    if data_config['source'] == 'huggingface':
        dataset = load_dataset(data_config['name'])
    else:
        dataset = load_dataset(data_config['source'], data_dir=data_config['name'])

    print(f"Using dataset: {data_config['name']}\n")

    # Get transforms
    train_transforms, base_transforms = get_transforms(data_config)

    # Prepare datasets
    processed_datasets = {}
    for split in dataset.keys():
        is_train = split == 'train'
        transforms_to_apply = train_transforms if is_train else base_transforms
        processed_datasets[split] = dataset[split].with_transform(
            lambda example: transform_and_encode(example, processor, transforms_to_apply, include_hand_features)
        )
    
    extra_test_datasets = {}
    for dataset_name, dataset_info in data_config.get('additional_test_datasets', {}).items():
        if dataset_info['source'] == 'huggingface':
            dataset = load_dataset(dataset_info['name'])
        else:
            dataset = load_dataset(dataset_info['source'], data_dir=dataset_info['name'])

        # Only process the test split for additional datasets
        if 'test' in dataset:
            extra_test_datasets[dataset_name] = dataset['test'].with_transform(
                lambda example: transform_and_encode(example, processor, base_transforms, include_hand_features)
            )

    return processed_datasets, extra_test_datasets

def get_dataset_splits(processed_datasets):
    train_dataset = processed_datasets['train']
    eval_dataset = processed_datasets['validation']
    test_dataset = processed_datasets['test']

    return train_dataset, eval_dataset, test_dataset
