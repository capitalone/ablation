import numpy as np

from . import distributions
from .distributions import (
    constant,
    constant_mean,
    constant_median,
    gaussian,
    gaussian_blur,
    marginal,
    max_distance,
)

PERTURBATIONS = [
    "gaussian_blur",
    "constant",
    "constant_mean",
    "constant_median",
    "gaussian",
    "marginal",
    "max_distance",
]


def generate_perturbation_distribution(
    method: str, X: np.ndarray, X_obs: np.ndarray, random_state=42, **kwargs
) -> np.ndarray:
    """Generate perturbation distribution to be used for ablation

    Args:
        method (str): Perturbation method
        X (np.ndarray): Data for source distribution
        X_obs (np.ndarray): Data observations to perturb
        random_state (Optional[int]): random seed


    Returns:
        np.ndarray: Perturbed dataset
    """
    np.random.seed(random_state)

    if method == "gaussian_blur":
        perturbation = gaussian_blur(X=X_obs, **kwargs)
    elif method == "constant":
        perturbation = constant(X=X_obs, **kwargs)
    elif method == "constant_mean":
        perturbation = constant_mean(X=X, **kwargs)
    elif method == "constant_median":
        perturbation = constant_median(X=X, **kwargs)
    elif method == "max_distance":
        perturbation = max_distance(X=X, X_obs=X_obs, **kwargs)
    elif method == "gaussian":
        perturbation = gaussian(X=X_obs, **kwargs)
    elif method == "marginal":
        perturbation = marginal(X=X, X_obs=X_obs, **kwargs)
    else:
        raise ValueError(f"Perturbation method '{method}' does not exist!")

    if method in distributions.CONSTANT:
        return np.tile(perturbation, (len(X_obs), 1))

    return perturbation
