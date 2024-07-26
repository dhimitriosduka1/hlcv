import numpy as np
import torch
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import Trainer

from common.utils import preds_or_target_to_tensor

"""Computes the mean average precision at different intersection over union (IoU) thresholds, 
specifically at 0.5."""
mean_average_precision_50 = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    iou_thresholds=[0.5],
    extended_summary=True,
)

"""Computes the mean average precision at different intersection over union (IoU) thresholds,
specifically averaging over the IoU thresholds from 0.5 to 0.95, with steps of 0.05."""
mean_average_precision_50_95 = MeanAveragePrecision(
    box_format="xyxy", iou_type="bbox", extended_summary=True
)


def compute_metrics(eval_pred: tuple[list[dict], list[dict]]) -> dict:
    """Returns a `dict` containing the mean average precision at 0.5, the mean average precision
    from 0.5 to 0.95, the precision, and the recall."""
    predictions, labels = eval_pred

    predictions = preds_or_target_to_tensor(predictions)
    labels = preds_or_target_to_tensor(labels)

    ap50_95 = mean_average_precision_50_95(predictions, labels)
    ap50 = mean_average_precision_50(predictions, labels)

    precision = ap50_95["precision"]
    recall = ap50_95["recall"]

    precision = ap50_95["precision"]
    recall = ap50_95["recall"]

    precision = precision[precision >= 0].mean().item()
    recall = recall[recall >= 0].mean().item()
    ap50_95 = ap50_95["map"].item()
    ap50 = ap50["map"].item()

    return {"mAP50": ap50, "mAP50-95": ap50_95, "precision": precision, "recall": recall}


def process_metrics(metrics: dict) -> tuple:
    """Processes a metric in a value of the `dict` returned by the `compute_metrics()` function with
    regards to "mAP50" and "mAP50-95", and returns the precision, recall, f1 score, and scores,
    needed for plotting the Precision-Recall curve, Recall-Confidence curve, Precision-Confidence
    curve, and F1-confidence curve."""
    precision = metrics["precision"].detach().clone()  # (T, R, K, A, M)
    scores = metrics["scores"].detach().clone()  # (T, R, K, A, M)

    # Process Precision-Recall curve data
    precision[precision < 0] = torch.nan
    precision = precision.nanmean(axis=(0, 2, 3, 4))  # (R,)

    # The recall values are implicitly [0, 0.01, 0.02, ..., 0.99, 1]
    recall = np.linspace(0, 1, 101)

    # Process Recall-Confidence curve data
    scores[scores < 0] = torch.nan
    scores = scores.nanmean(axis=(0, 2, 3, 4)).flatten()

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return precision, recall, f1, scores


def log_table_data(
    trainer: Trainer,
    eval_dataset=None,
    use_eval=True,
    per_device_eval_batch_size=None,
    verbose=True,
):
    """Logs Precision-Recall, Precision-Confidence, Recall-Confidence, and F1-Confidence tables
    to wandb. It uses the `trainer` to predict the dataset. If `use_eval` is True, it uses the
    evaluation dataset saved on the `trainer`. Otherwise, it uses the test dataset."""
    if per_device_eval_batch_size is not None:
        trainer.args.per_device_eval_batch_size = per_device_eval_batch_size

    # Evaluate the model to get predictions
    if eval_dataset is not None:
        eval_results = trainer.predict(eval_dataset)
    elif use_eval:
        eval_results = trainer.predict(trainer.eval_dataset)
    else:
        eval_results = trainer.predict(trainer.test_dataset)

    predictions = eval_results[0]
    labels = eval_results[1]

    if verbose:
        print(f"Results: {eval_results[2]}")

    predictions = preds_or_target_to_tensor(predictions)
    labels = preds_or_target_to_tensor(labels)

    # Compute the mean average precision at 50 IoU
    ap50 = mean_average_precision_50(predictions, labels)

    # Get the precision, recall, and scores arrays
    pr_precision, pr_recall, f1, pr_scores = process_metrics(ap50)

    # Now use these processed arrays to create wandb tables
    precision_recall_data = [["Guitar-necks", p, r] for p, r in zip(pr_precision, pr_recall)]
    precision_confidence_data = [["Guitar-necks", p, c] for p, c in zip(pr_precision, pr_scores)]
    recall_confidence_data = [["Guitar-necks", r, c] for r, c in zip(pr_recall, pr_scores)]
    f1_confidence_data = [["Guitar-necks", f, c] for f, c in zip(f1, pr_scores)]

    # Create wandb.Tables
    precision_recall_table = wandb.Table(columns=["class", "y", "x"], data=precision_recall_data)
    precision_confidence_table = wandb.Table(
        columns=["class", "y", "x"], data=precision_confidence_data
    )
    recall_confidence_table = wandb.Table(columns=["class", "y", "x"], data=recall_confidence_data)
    f1_confidence_table = wandb.Table(columns=["class", "y", "x"], data=f1_confidence_data)

    # Log tables to wandb
    wandb.log(
        {
            "curves/Precision-Recall(B)_table": precision_recall_table,
            "curves/Precision-Confidence(B)_table": precision_confidence_table,
            "curves/Recall-Confidence(B)_table": recall_confidence_table,
            "curves/F1-Confidence(B)_table": f1_confidence_table,
        }
    )
