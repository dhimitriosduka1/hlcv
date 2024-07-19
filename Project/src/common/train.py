# train.py
import argparse
import os
from transformers import Trainer, TrainingArguments
from config import load_config
from data_processing import load_and_prepare_dataset, get_dataset_splits
from model import load_model_and_processor
import wandb
from transformers.integrations import WandbCallback

def main(config_path):
    # Load configuration
    config = load_config(config_path)

    wandb_config = load_config("Project/src/common/wandb.yml")

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
    processed_datasets = load_and_prepare_dataset(config['data'], config['model'], processor)
    train_dataset, eval_dataset, test_dataset = get_dataset_splits(processed_datasets)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        logging_dir=config['training']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        report_to="wandb",
        metric_for_best_model="accuracy",
        learning_rate=config['training']['learning_rate']
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbCallback()],
    )

    # Start training
    # trainer.train()

    # Save the final model
    trainer.save_model(config['training']['final_model_path'])

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
    args = parser.parse_args()
    main(args.config)