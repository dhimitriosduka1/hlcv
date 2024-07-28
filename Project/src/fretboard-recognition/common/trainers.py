from itertools import accumulate
from transformers import Trainer
from common.utils import safe_item


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_total_loss = True
        self.__reset_loss_tracking()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if isinstance(outputs, dict):
            # This means that it came from the evaluation loop, otherwise from the training loop
            if "logits" in outputs:
                self.total_eval_loss += loss
                self.eval_cls_loss += outputs.get("cls_loss", 0.0)
                self.eval_box_loss += outputs.get("box_loss", 0.0)
                if not self.report_total_loss:
                    self.num_eval_losses += 1
            else:
                self.total_train_loss += loss
                self.train_cls_loss += outputs.get("cls_loss", 0.0)
                self.train_box_loss += outputs.get("box_loss", 0.0)
                if not self.report_total_loss:
                    self.num_train_losses += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs):
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs}

        # Log training losses
        if "loss" in output:
            output["train_loss"] = safe_item(self.total_train_loss) / self.num_train_losses
        if self.train_cls_loss is not None:
            output["train_cls_loss"] = safe_item(self.train_cls_loss) / self.num_train_losses
        if self.train_box_loss is not None:
            output["train_box_loss"] = safe_item(self.train_box_loss) / self.num_train_losses

        # Log evaluation losses
        if "eval_loss" in output:
            output["eval_loss"] = safe_item(self.total_eval_loss) / self.num_eval_losses
        if self.eval_cls_loss is not None:
            output["eval_cls_loss"] = safe_item(self.eval_cls_loss) / self.num_eval_losses
        if self.eval_box_loss is not None:
            output["eval_box_loss"] = safe_item(self.eval_box_loss) / self.num_eval_losses

        # Reset loss tracking for the next epoch
        self.__reset_loss_tracking()
        super().log(output)

    def __reset_loss_tracking(self) -> None:
        self.total_train_loss = 0.0
        self.total_eval_loss = 0.0
        self.train_cls_loss = 0.0
        self.train_box_loss = 0.0
        self.eval_cls_loss = 0.0
        self.eval_box_loss = 0.0
        self.num_train_losses = 1.0 if self.report_total_loss else 0.0
        self.num_eval_losses = 1.0 if self.report_total_loss else 0.0
