import torch
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict
from datasets import load_metric
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

metric = load_metric("seqeval")


def compute_metrics(predictions: torch.Tensor, labels: List[List[int]], label_names: List[str]) -> Dict:
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_names[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results


def compute_clf_metrics(predictions: torch.Tensor, labels: List[int], label_names: List[str]) -> Dict:
    predictions = np.argmax(predictions, axis=1)
    overall_accuracy = accuracy_score(labels, predictions)
    overall_fscore = f1_score(labels, predictions, average="macro")
    overall_precision = precision_score(labels, predictions, average="macro")
    overall_recall = recall_score(labels, predictions, average="macro")
    detailed_metrics = classification_report(
        labels, predictions, target_names=label_names, output_dict=True
    )
    return dict(
        overall_accuracy=overall_accuracy,
        overall_f1=overall_fscore,
        overall_precision=overall_precision,
        overall_recall=overall_recall,
        detailed_metrics=detailed_metrics,
    )


@dataclass()
class ClassifierOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    def __repr__(self):
        kws = [f"{key}={type(value).__name__}" for key, value in self.__dict__.items()]
        return f"{type(self).__name__}({', '.join(kws)})"

    def __str__(self):
        return self.__repr__()


@dataclass()
class NewClassifierOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = None
    fale_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    fake_logits: torch.FloatTensor = None
    probs: torch.FloatTensor = None
    fake_probs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    def __repr__(self):
        kws = [f"{key}={type(value).__name__}" for key, value in self.__dict__.items()]
        return f"{type(self).__name__}({', '.join(kws)})"

    def __str__(self):
        return self.__repr__()
