import os
import wandb
import argparse
import datetime

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.integrations import WandbCallback
from config import load_config
from metric_util import compute_metrics
from data_processing import load_and_prepare_dataset, get_dataset_splits
from model import load_model_and_processor
from collate_util import collate_fn

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

    # Load model and processor
    model, processor = load_model_and_processor(config['model'])

    # Load and prepare dataset
    processed_datasets = load_and_prepare_dataset(config['data'], processor)
    train_dataset, eval_dataset, test_dataset = get_dataset_splits(processed_datasets)

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
        eval_steps=10,
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
        early_stopping_patience=config['training'].get('early_stopping_patience', 3)
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
    trainer.save_model(
        f"{config['training']['final_model_path']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    )

    # Evaluate on test set if available
    if test_dataset:
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")
        wandb.log({"test": test_results})

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model for image classification")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--wandb", type=str, required=True, help="Path to wandb configuration file")
    args = parser.parse_args()
    main(args)
