from itertools import accumulate
from transformers import Trainer
from common.utils import count_parameters, estimate_gflops, measure_inference_speed, safe_item


class CustomTrainer(Trainer):
    def __init__(self, *args, report_total_loss=True, log_performance_metrics=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_total_loss = report_total_loss
        self.log_performance_metrics = log_performance_metrics
        self.model_performance_analyzed = False
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

    def model_performance_analysis(self, verbose=True) -> dict:
        """Analyzes the model's performance in terms of the number of parameters, the number of
        GigaFLOPs, and the inference speed in milliseconds.

        Returns
        -------
        dict
            A dictionary with keys being "model/parameters", "model/GFLOPs", and
            "model/speed_PyTorch(ms)".
        """
        # Put model in eval mode for analysis
        self.model.eval()

        # Perform model performance analysis
        num_params = count_parameters(self.model)
        gflops = estimate_gflops(self.model)
        inference_speed = measure_inference_speed(self.model)

        if verbose:
            print(f"Model Performance:")
            print(f"Parameters: {num_params}")
            print(f"GFLOPs: {gflops:.2f}")
            print(f"Inference Speed: {inference_speed:.2f} ms")

        return {
            "model/parameters": num_params,
            "model/GFLOPs": gflops,
            "model/speed_PyTorch(ms)": inference_speed,
        }

    def __reset_loss_tracking(self) -> None:
        self.total_train_loss = 0.0
        self.total_eval_loss = 0.0
        self.train_cls_loss = 0.0
        self.train_box_loss = 0.0
        self.eval_cls_loss = 0.0
        self.eval_box_loss = 0.0
        self.num_train_losses = 1.0 if self.report_total_loss else 0.0
        self.num_eval_losses = 1.0 if self.report_total_loss else 0.0
