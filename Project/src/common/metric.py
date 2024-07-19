import numpy as np
from datasets import load_metric
from metric import compute_metrics
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    metrics = compute_metrics(eval_pred)
    predictions = np.argmax(eval_pred.predictions, axis=1)
    accuracy = accuracy_score(eval_pred.label_ids, predictions)
    metrics['accuracy'] = accuracy
    return metrics