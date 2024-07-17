#!/usr/bin/env python

# %%
# For Google Colab
# !pip install roboflow
# !pip install -U transformers
# !pip install datasets
# !pip install wandb
# !pip install accelerate -U

# %%
import yaml

import wandb
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets import load_dataset, DatasetDict, load_metric
from transformers import ViTForImageClassification, TrainingArguments, Trainer, AutoImageProcessor, ResNetForImageClassification

# %%
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# %%
def create_transform(aug_config):
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

    # Always include resizing
    transform_list.extend([
        transforms.Resize((224, 224)),
    ])

    return transforms.Compose(transform_list)

# %%
def get_transforms(config):
    return create_transform(config['data']['train_augmentation']), create_transform(config['data'].get('val_augmentation', {}))

# %%
f_run_config = "config-vit.yml"
f_wandb_config = "wandb.yml"

# %%
# Load configuration
config = load_config(f_run_config)
wandb_config = load_config(f_wandb_config)

# %%
# Get transforms
train_transform, base_transform = get_transforms(config)

# %%
# Load the ds
ds = load_dataset("dduka/guitar-chords")

# %%
labels = ds['train'].features['label']

# %%
model_name = config['model']['name']

if model_name == 'vit':
    model = ViTForImageClassification.from_pretrained(
        config['model']['pretrained_weights'],
        num_labels=len(labels.names),
        id2label={str(i): c for i, c in enumerate(labels.names)},
        label2id={c: str(i) for i, c in enumerate(labels.names)},
        ignore_mismatched_sizes=True
    )
elif model_name == 'resnet152':
    model = ResNetForImageClassification.from_pretrained(
        config['model']['pretrained_weights'],
        num_labels=len(labels.names)
    )
else:
    raise ValueError(f"Model {model_name} not supported")

processor = AutoImageProcessor.from_pretrained(config['model']['pretrained_weights'])

# %%
def transform(batch, is_train=True):
    # Resize the images to the desired size
    train_transforms, base_transforms = get_transforms(config)
    if is_train:
        processed_images = [train_transforms(x.convert("RGB")) for x in batch['image']]
    else:
        processed_images = [base_transforms(x.convert("RGB")) for x in batch['image']]

    inputs = processor(processed_images, return_tensors='pt')
    inputs['label'] = batch['label']

    return inputs

# %%
# Split the data
ds = DatasetDict({
    'train': ds['train'].with_transform(lambda batch: transform(batch, True)),
    'test': ds['test'].with_transform(lambda batch: transform(batch, False)),
    'valid': ds['validation'].with_transform(lambda batch: transform(batch, False))
})

# %%
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

metric = load_metric("accuracy")

# %%
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# %%
# Initialize wandb
wandb.require("core")
wandb.init(
    project=wandb_config["project"],
    name=wandb_config['name'] + "-" + wandb.util.generate_id(),
    config=wandb_config,
    entity=wandb_config["entity"]
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=config['training']['output_dir'],
    num_train_epochs=config['training']['num_epochs'],
    per_device_train_batch_size=config['training']['batch_size'],
    per_device_eval_batch_size=config['training']['batch_size'],
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=float(config['training']['learning_rate']),
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="wandb",
    remove_unused_columns=False,
    logging_steps=500,
    save_total_limit=1,
    # fp16=True
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=ds["train"],
    eval_dataset=ds["valid"],
    compute_metrics=compute_metrics,
    tokenizer=processor
)

# Train the model
trainer.train()

# # Save the fine-tuned model
trainer.save_model(config['training']['final_model_path'])

# %%
metrics = trainer.evaluate(ds['test'])
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

test_ds = ds['test']
test_outputs = trainer.predict(test_ds)

y_true = test_outputs.label_ids
y_pred = test_outputs.predictions.argmax(1)

labels = test_ds.features["label"].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)

recall = recall_score(y_true, y_pred, average=None)

# Print the recall for each class
for label, score in zip(labels, recall):
  print(f"Recall for {label}: {score:.2f}")

# %%
# Close wandb run
wandb.finish()


