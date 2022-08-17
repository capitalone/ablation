from typing import Callable, Union

import numpy as np
import torch
from sklearn import metrics
from sklearn.base import ClassifierMixin

from .model import _predict_proba_model_type


def abs_diff(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Absolute probability difference

    Args:
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): probabilities

    Returns:
        float: absolute prob difference
    """
    if len(y_pred.shape) == 2:
        return np.mean(1 - y_pred[np.arange(len(y_pred)), y_true])

    return np.abs(y_true - y_pred).mean()

    # def _get_model_performance(self) -> np.ndarray:
    #     """Return performance for a model's predicted probabilities

    #     Returns:
    #         np.ndarray: aupr or auroc for predictions
    #     """

    #     y_pred = _predict_proba_model_type(self.X, self.model)
    #     if self.scoring_method == "log_loss":
    #         return metrics.log_loss(self.y, y_pred)
    #     if self.scoring_method == "abs_diff":
    #         return abs_diff(self.y, y_pred)
    #     if self.scoring_method == "auroc":
    #         return metrics.roc_auc_score(self.y, y_pred, multi_class="ovr")

    #     return metrics.average_precision_score(self.y, y_pred, average="macro")


def eval_model_performance(
    model: Union[torch.nn.Module, ClassifierMixin, Callable],
    X: np.ndarray,
    y_true: np.ndarray,
    scoring_method="log_loss",
) -> float:
    """[summary]

    Args:
        model (Union[torch.nn.Module, ClassifierMixin, Callable]): model
        X (np.ndarray): dataset to evaluate
        y_true (np.ndarray): true labels
        scoring_method (str, optional): Scoring method ('log_loss','abs_diff', 'auroc', 'aupr'). Defaults to "log_loss".

    Returns:
        float: model performance
    """
    y_pred = _predict_proba_model_type(X, model)
    if scoring_method == "log_loss":
        return metrics.log_loss(y_true, y_pred)
    if scoring_method == "abs_diff":
        return abs_diff(y_true, y_pred)
    if scoring_method == "auroc":
        return metrics.roc_auc_score(y_true, y_pred, multi_class="ovr")

    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        return metrics.average_precision_score(
            np.eye(y_pred.shape[1])[y_true], y_pred, average="macro"
        )
    return metrics.average_precision_score(y_true, y_pred, average="macro")


def append_dict_lists(a, b):
    for k in a:
        a[k].append(b[k])
    return a
