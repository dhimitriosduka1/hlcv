import torch
import wandb
import torch
import argparse
import datetime
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from config import load_config
from metric_util import compute_metrics
from data_processing import load_and_prepare_dataset, get_dataset_splits
from model import load_model, load_processor
from collate_util import collate_fn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class HybridModel(torch.nn.Module):
    def __init__(self, chord_model, hand_pose_model, num_classes):
        super().__init__(HybridModel)
        self.chord_model = chord_model
        self.hand_pose_model = hand_pose_model
        self.classifier = torch.nn.Linear(chord_model.config.hidden_size + hand_pose_model.config.hidden_size, num_classes)

        # Freeze the hand pose model
        for param in self.hand_pose_model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, hand_pose_pixel_values):
        chord_outputs = self.chord_model(pixel_values)
        with torch.no_grad():   
            hand_pose_outputs = self.hand_pose_model(hand_pose_pixel_values)
        
        combined_features = torch.cat((chord_outputs.pooler_output, hand_pose_outputs.pooler_output), dim=1)
        logits = self.classifier(combined_features)
        return logits

def combined_collate_fn(batch):
    chord_inputs = [item['pixel_values'] for item in batch]
    hand_pose_inputs = [item['hand_pose_pixel_values'] for item in batch]
    labels = [item['label'] for item in batch]
    
    chord_inputs = torch.stack(chord_inputs)
    hand_pose_inputs = torch.stack(hand_pose_inputs)
    labels = torch.tensor(labels)
    
    return {
        'pixel_values': chord_inputs,
        'hand_pose_pixel_values': hand_pose_inputs,
        'labels': labels
    }

def main(args):
    EVAL_AND_SAVE_STEPS = 10
    STRATEGY = "steps"
    SAVE_TOTAL_LIMIT = 2
    LOGING_STEPS = 100
    LOGING_DIR = "../../logs"
    OUTPUT_DIR = "./output"
    METRIC_FOR_BEST_MODEL = "accuracy"

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

    # Load labels
    labels = train_dataset.features["label"].names

    chord_model = load_model(config['model'], labels)
    hand_pose_model = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    hybrid_model = HybridModel(chord_model, hand_pose_model, len(labels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hybrid_model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        use_cpu=False,
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        remove_unused_columns=False,
        output_dir=OUTPUT_DIR,
        logging_dir=LOGING_DIR,
        logging_steps=LOGING_STEPS,
        eval_strategy=STRATEGY,
        save_strategy=STRATEGY,
        eval_steps=EVAL_AND_SAVE_STEPS,
        save_steps=EVAL_AND_SAVE_STEPS,
        logging_strategy=STRATEGY,

        # Early stopping 
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        learning_rate=float(config['training']['learning_rate']),
        save_total_limit=SAVE_TOTAL_LIMIT,
        greater_is_better=True,
        
        # WANDB
        report_to="wandb",
    )

    # Initialize EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training'].get('early_stopping_patience', 5),
        early_stopping_threshold=0.01
    )

    # Initialize Trainer
    trainer = Trainer(
        model=hybrid_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stopping_callback],
        compute_metrics=compute_metrics,
        data_collator=combined_collate_fn,
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

    # Evaluate on additional test datasets
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
