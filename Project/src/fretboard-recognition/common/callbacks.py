import wandb
from transformers import TrainerCallback


class CustomWandbCallback(TrainerCallback):
    """A custom Weights & Biases callback for the Hugging Face Trainer. This callback logs the
    following metrics:

    ```
    train/loss
    train/cls_loss
    train/box_loss
    eval/loss
    eval/cls_loss
    eval/box_loss
    metrics/precision(B)
    metrics/recall(B)
    metrics/mAP50(B)
    metrics/mAP50-95(B)
    ```
    """

    def __init__(self):
        self.log_dict = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        log_dict = {}

        # Log training metrics
        if "train_loss" in logs:
            log_dict = {
                **log_dict,
                "train/loss": logs["train_loss"],
                "train/cls_loss": logs.get("train_cls_loss", 0),
                "train/box_loss": logs.get("train_box_loss", 0),
            }

        # Log evaluation metrics
        if "eval_loss" in logs:
            log_dict = {
                **log_dict,
                "val/loss": logs["eval_loss"],
                "val/cls_loss": logs.get("eval_cls_loss", 0),
                "val/box_loss": logs.get("eval_box_loss", 0),
            }

        # Log other evaluation metrics
        if "eval_mAP50" in logs:
            log_dict = {
                **log_dict,
                "metrics/precision(B)": logs.get("eval_precision", 0),
                "metrics/recall(B)": logs.get("eval_recall", 0),
                "metrics/mAP50(B)": logs.get("eval_mAP50", 0),
                "metrics/mAP50-95(B)": logs.get("eval_mAP50-95", 0),
            }

        self.log_dict = {**self.log_dict, **log_dict}

        if (
            "train/loss" in self.log_dict
            and "val/loss" in self.log_dict
            and "metrics/mAP50(B)" in self.log_dict
        ):
            wandb.log(self.log_dict, step=int(state.epoch))
            self.log_dict = {}