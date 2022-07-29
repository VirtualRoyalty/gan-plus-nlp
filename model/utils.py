import torch
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict
from datasets import load_metric

metric = load_metric("seqeval")


def compute_metrics(predictions: torch.Tensor,
                    labels: List[List[int]],
                    label_namess: List[str]
                    ) -> Dict:
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_namess[pred] for (pred, _label) in zip(prediction, _label) if _label != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_namess[_label] for (pred, _label) in zip(prediction, _label) if _label != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results


@dataclass()
class TokenClassifierOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    def __repr__(self):
        kws = [f"{key}={type(value).__name__}"
               for key, value in self.__dict__.items()]
        return f"{type(self).__name__}({', '.join(kws)})"

    def __str__(self):
        return self.__repr__()