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
        wandb.define_metric("Step")
        wandb.define_metric("metrics/precision(B)", step_metric="Step")

    def on_log(self, args, state, control, logs=None, **kwargs):
        log_dict = {}

        # Log training metrics
        if "loss" in logs:
            log_dict = {
                **log_dict,
                "train/loss": logs["loss"],
                "train/cls_loss": logs.get("cls_loss", 0),
                "train/box_loss": logs.get("box_loss", 0),
            }

        # Log evaluation metrics
        if "eval_loss" in logs:
            log_dict = {
                **log_dict,
                "eval/loss": logs["eval_loss"],
                "eval/cls_loss": logs.get("eval_cls_loss", 0),
                "eval/box_loss": logs.get("eval_box_loss", 0),
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
            and "eval/loss" in self.log_dict
            and "metrics/mAP50(B)" in self.log_dict
        ):
            self.log_dict = {**self.log_dict, "Step": int(state.epoch)}
            wandb.log(self.log_dict)
            self.log_dict = {}