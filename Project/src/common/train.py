import os
import wandb
import torch
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.integrations import WandbCallback
from config import load_config
from metric_util import compute_metrics
from data_processing import load_and_prepare_dataset, get_dataset_splits
from model import load_model, load_processor
from collate_util import collate_fn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

def main(args):
    # Load configurations
    config = load_config(args.config)
    wandb_config = load_config(args.wandb)

    # Initialize wandb
    wandb.init(
        project=wandb_config['project'],
        name=config['wandb']['run_name'],
        entity=wandb_config['entity'],
        config=config
    )

    # Load the processor
    processor = load_processor(config['model'])

    # Load and prepare dataset
    processed_datasets, additional_test_datasets = load_and_prepare_dataset(config['data'], processor)
    train_dataset, eval_dataset, test_dataset = get_dataset_splits(processed_datasets)

    # Append current test ds to additional_test_datasets 
    additional_test_datasets["current_run"] = test_dataset

    # Load model and processor
    labels = train_dataset.features["label"].names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(config['model'], labels)
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        logging_dir=config['training']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        report_to="wandb",
        metric_for_best_model="accuracy",
        learning_rate=float(config['training']['learning_rate']),
        save_total_limit=2,
        greater_is_better=True,
        logging_strategy="steps"
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training'].get('early_stopping_patience', 3),
        early_stopping_threshold=0.01
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stopping_callback],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        tokenizer=processor
    )

    # Start training
    trainer.train()

    # Save the final model
    model_dir = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    model_path = f"../models/{model_dir}/{config['training']['final_model_path']}"
    model.save_pretrained(model_path, from_pt=True) 

    print("-" * 100)
    print(f"model_path:{ model_path}")
    print("-" * 100)

    for dataset_name, test_dataset in additional_test_datasets.items():
        print(f"Evaluating on {dataset_name}")
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results for {dataset_name}: {test_results}")
        wandb.log({f"test_{dataset_name}": test_results})
        
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, xticks_rotation=45)
        plt.title(f'Confusion Matrix - {dataset_name}')

        wandb.log({f"confusion_matrix_{dataset_name}": wandb.Image(plt)})

        plt.close()

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model for image classification")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--wandb", type=str, required=True, help="Path to wandb configuration file")
    args = parser.parse_args()
    main(args)
