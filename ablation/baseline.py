from typing import Optional

import numpy as np

from . import distributions
from .distributions import (
    constant,
    constant_mean,
    constant_median,
    gaussian,
    gaussian_blur,
    gaussian_blur_permutation,
    max_distance,
    nearest_neighbors,
    nearest_neighbors_counterfactual,
    opposite_class,
    training,
)
from .utils.general import sample
from .utils.transform import le_to_ohe, ohe_to_le

# from shap import sample


BASELINES = [
    "gaussian_blur",
    "gaussian_blur_permutation",
    "constant",
    "constant_mean",
    "constant_median",
    "gaussian",
    "training",
    "max_distance",
    "nearest_neighbors",
    "nearest_neighbors_counterfactual",
    "opposite_class",
]


class OneToOneBaseline(np.ndarray):
    pass


class ManyToOneBaseline(np.ndarray):
    pass


class SampleBaseline(np.ndarray):
    pass


class ConstantBaseline(np.ndarray):
    pass


def generate_baseline_distribution(
    method: str,
    X: np.ndarray,
    X_obs: np.ndarray,
    y: Optional[np.ndarray] = None,
    y_obs: Optional[np.ndarray] = None,
    nsamples: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate baseline distribution for explanations

    Args:
        method (str): baseline method
        X (np.ndarray): data for source distribution
        X_obs (np.ndarray): data observations to explain with baselines
        y (Optional[np.ndarray]): classes of source distribution
        y_obs: (Optional[np.ndarray]): predicted classes of observations to explain
        nsamples (Optional[int]): number of samples
        random_state (Optional[int]): random seed

    Returns:
        np.ndarray: baseline
    """

    # Signal to distribution functions that they will be used for
    # baseline generation.
    kwargs["baseline"] = True

    np.random.seed(random_state)

    if nsamples is None and method in distributions.SAMPLE:
        raise ValueError(f"nsamples cannot be None for method: {method}")

    if method == "gaussian_blur":
        baseline = gaussian_blur(X, **kwargs)
    elif method == "gaussian_blur_permutation":
        baseline = gaussian_blur_permutation(X, **kwargs)
    elif method == "constant":
        baseline = constant(X, **kwargs)
    elif method == "constant_mean":
        baseline = constant_mean(X, **kwargs)
    elif method == "constant_median":
        baseline = constant_median(X, **kwargs)
    elif method == "gaussian":
        baseline = gaussian(X, **kwargs)
    elif method == "training":
        baseline = training(X, **kwargs)
    elif method == "max_distance":
        baseline = max_distance(X, X_obs, **kwargs)
    elif method == "nearest_neighbors":
        baseline = nearest_neighbors(X, X_obs, **kwargs)
    elif method == "nearest_neighbors_counterfactual":
        baseline = nearest_neighbors_counterfactual(
            X, y, X_obs, y_obs, **kwargs
        )
    elif method == "opposite_class":
        baseline = opposite_class(X, y, y_obs, nsamples, **kwargs)
    else:
        raise ValueError(f"Baseline method '{method}' does not exist!")

    if method in distributions.SAMPLE:
        baseline = sample(baseline, nsamples, random_state)
        return baseline.view(SampleBaseline)
    elif method in distributions.MANY_TO_ONE:
        return baseline.view(ManyToOneBaseline)
    elif method in distributions.CONSTANT:
        return baseline.view(ConstantBaseline)
    return baseline.view(OneToOneBaseline)
