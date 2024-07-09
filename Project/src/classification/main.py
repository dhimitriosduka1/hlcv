import os
import io
import stat
import yaml
import wandb
import torch
import shutil
import zipfile
import requests
from roboflow import Roboflow
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms, datasets
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer

def download_roboflow_data(config):
    """
    Download dataset from RoboFlow.
    """
    roboflow_config = config['data']['roboflow']
    roboflow = Roboflow(api_key=roboflow_config["api_key"])
    project = roboflow.workspace(roboflow_config["workspace"]).project(roboflow_config["project"])
    version = project.version(roboflow_config["version"])
    dataset = version.download(model_format=roboflow_config["version_download"])

    os.makedirs(config['data']['path'], exist_ok=True)
    shutil.move(src=dataset.location, dst=config['data']['path'])

    print(f"Dataset downloaded and extracted to {config['data']['path']}")

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_transform(aug_config, processor):
    transform_list = []
    
    # Add transforms based on configuration
    if 'random_resize_crop' in aug_config:
        transform_list.append(transforms.RandomResizedCrop(**aug_config['random_resize_crop']))
    if 'random_horizontal_flip' in aug_config:
        transform_list.append(transforms.RandomHorizontalFlip(aug_config['random_horizontal_flip']))
    if 'color_jitter' in aug_config:
        transform_list.append(transforms.ColorJitter(**aug_config['color_jitter']))
    if 'random_rotation' in aug_config:
        transform_list.append(transforms.RandomRotation(aug_config['random_rotation']))
    
    # Always include resizing, ToTensor, and normalization
    transform_list.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    
    return transforms.Compose(transform_list)

def get_transforms(config, processor):
    train_transform = create_transform(config['data']['train_augmentation'], processor)
    val_transform = create_transform(config['data']['val_augmentation'], processor)
    
    return train_transform, val_transform

def load_data(data_dir, transform):
    return datasets.ImageFolder(data_dir, transform=transform)

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Download data from RoboFlow if specified
    if config['data'].get('use_roboflow', False):
        download_roboflow_data(config)
    
    return

    # Initialize wandb
    wandb.init(project=config['wandb']['project_name'], config=config)
    
    # Load pre-trained model and processor
    model = ViTForImageClassification.from_pretrained(config['model']['pretrained_weights'])
    processor = ViTImageProcessor.from_pretrained(config['model']['pretrained_weights'])
    
    # Get transforms
    train_transform, val_transform = get_transforms(config, processor)

    full_dataset = ChordClassificationDataset(config['data']['data_dir'], transform=None)
    
    # Split the dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform  # Using val_transform for test set as well
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config['training']['learning_rate'],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb",
    )
    
    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()},
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(config['training']['final_model_path'])
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main("/home/dhimitriosduka/Documents/UdS/SoSe 2024/High-Level Computer Vision/Assignments/hlcv/Project/src/classification/config.yml")