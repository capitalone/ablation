from functools import partial
from typing import Callable, Union

import numpy as np
import torch
from sklearn.base import ClassifierMixin


def _torch_long(x: np.array, device=None) -> torch.tensor:
    return torch.tensor(x, device=device).long()


def _torch_float(x: np.array, device=None) -> torch.tensor:
    return torch.tensor(x, device=device).float()


def _as_numpy(x: torch.tensor) -> np.ndarray:
    return x.cpu().detach().numpy().astype(float)


def _predict_proba_model_type(
    X: np.array, model: Union[torch.nn.Module, ClassifierMixin, Callable],
) -> np.ndarray:
    """Checks type of model and predicts probability accordingly

    Args:
        X (np.array): numpy array
        model (Union[torch.nn.Module, ClassifierMixin, Callable]): model

    Raises:
        ValueError: if model type is unsupported

    Returns:
        np.ndarray: prediction
    """
    if isinstance(model, ClassifierMixin):
        pred = model.predict_proba(X)
    elif isinstance(model, torch.nn.Module):
        pred = _as_numpy(model(_torch_float(X)))
    elif isinstance(model, Callable):
        pred = model(X)
    else:
        raise ValueError("Model type not supported")

    output_shape = pred.shape

    if len(output_shape) == 1:
        return pred
    elif output_shape[1] == 1:
        return pred.flatten()
    elif output_shape[1] == 2:
        return pred[:, 1]

    return pred


def _predict_model_type(
    X: np.array, model: Union[torch.nn.Module, ClassifierMixin, Callable],
) -> np.ndarray:
    """Checks type of model and predicts accordingly

    Args:
        X (np.array): numpy array
        model (Union[torch.nn.Module, ClassifierMixin, Callable]): model

    Returns:
        np.ndarray: prediction
    """

    pred = _predict_proba_model_type(X, model)
    output_shape = pred.shape

    if len(output_shape) == 1:
        return np.round(pred)

    return np.argmax(pred, -1)


def _predict_proba_fn(
    model: Union[torch.nn.Module, ClassifierMixin, Callable]
) -> Callable:
    """Generalize all model predict probability functions

    Args:
        model (Union[torch.nn.Module, ClassifierMixin, Callable]): model

    Returns:
        Callable: generalized predict proba function
    """
    return partial(_predict_proba_model_type, model=model)


def _predict_fn(
    model: Union[torch.nn.Module, ClassifierMixin, Callable]
) -> Callable:
    """Generalize all model predict functions

    Args:
        model (Union[torch.nn.Module, ClassifierMixin, Callable]): model

    Returns:
        Callable: generalized predict proba function
    """
    return partial(_predict_model_type, model=model)
