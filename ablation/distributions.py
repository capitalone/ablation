"""
perturb.py
About: Feature distributions for baselines and perturbations
"""
from typing import Optional

import numpy as np
from numpy.random import permutation, randn
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn import base
from sklearn.neighbors import NearestNeighbors

from .utils.general import sample
from .utils.transform import le_to_ohe, ohe_to_le

# Main differences between perturbations and baselines:
# Baselines -- will be smaller samples
# Perturbations -- size of test set

ONE_TO_ONE = ["max_distance"]
MANY_TO_ONE = [
    "nearest_neighbors",
    "nearest_neighbors_counterfactual",
    "opposite_class",
]
SAMPLE = [
    "gaussian",
    "gaussian_blur",
    "gaussian_blur_permutation",
    "training",
    "marginal",
]
CONSTANT = ["constant", "constant_mean", "constant_median"]


def categorical_perturbation_case(**kwargs) -> bool:
    """
    Check for categoricals that need special care
    during distribution generation.  The categoricals
    should only be treated this way during perturbation.
    Baselines should use the default format.

    Returns:
        bool: Boolean indicating if distribution should be
              handling categorical features another way.
    """
    if (
        ("agg_map" in kwargs)
        and ("baseline" not in kwargs)
        and (kwargs["agg_map"] != None)
    ):
        return True
    return False


def constant(
    X: np.ndarray, value: Optional[float] = 0.0, **kwargs
) -> np.ndarray:
    """Generate a constant distribution

    Args:
        X (np.ndarray): Data to get shape
        value (float, optional): Constant value. Defaults to 0.0.

    Returns:
        np.array: constant distribution
    """
    if categorical_perturbation_case(**kwargs):
        X_LE = ohe_to_le(X, kwargs["agg_map"])
        return np.ones((1, X_LE.shape[1])) * value

    return np.ones((1, X.shape[1])) * value


def constant_mean(X: np.ndarray, **kwargs) -> np.ndarray:
    """Generate a constant mean distribution (mean value per feature)

    Args:
        X (np.ndarray): data to derive mean

    Returns:
        np.array: constant mean distribution
    """
    if categorical_perturbation_case(**kwargs):
        X_LE = ohe_to_le(X, kwargs["agg_map"])
        return np.mean(X_LE, axis=0, keepdims=True)

    return np.mean(X, axis=0, keepdims=True)


def constant_median(X: np.ndarray, **kwargs) -> np.ndarray:
    """Generate a constant median distribution (median value per feature)

    Args:
        X (np.ndarray): data to derive median

    Returns:
        np.array: constant median distribution
    """
    if categorical_perturbation_case(**kwargs):
        # For this case, median is calculated over
        # numerical features and mode is calculated
        # over the categorical features.
        X_LE = ohe_to_le(X, kwargs["agg_map"])
        median_mode = np.zeros((1, X_LE.shape[1]))  # To keep dimensions
        for (idx, mapping) in enumerate(kwargs["agg_map"]):
            if len(mapping) > 1:
                median_mode[0][idx] = stats.mode(X_LE[:, idx], axis=0)[0][
                    0
                ].astype(int)
            else:
                median_mode[0][idx] = np.median(X_LE[:, idx], axis=0)
        return median_mode

    return np.median(X, axis=0, keepdims=True)


def max_distance(X: np.ndarray, X_obs: np.ndarray, **kwargs) -> np.ndarray:
    """Furthest valid data sample in L1 distance

    Args:
        X (np.ndarray): data to derive min/max
        X_obs (np.ndarray): data observations for which to generate max_distance

    Returns:
        np.array: furthest valid data samples by L1 distance
    """

    if categorical_perturbation_case(**kwargs):
        X_LE = ohe_to_le(X, kwargs["agg_map"])
        X_obs_LE = ohe_to_le(X_obs, kwargs["agg_map"])

        max_value = np.tile(X_LE.max(axis=0), (len(X_obs_LE), 1))
        min_value = np.tile(X_LE.min(axis=0), (len(X_obs_LE), 1))
        midpoint = (min_value + max_value) / 2

        # Maximum distance implemented for numericals.
        # Categoricals are uniformly sampled instead.
        modified_max_distance = np.zeros(
            (X_obs_LE.shape[0], X_obs_LE.shape[1])
        )

        for (idx, mapping) in enumerate(kwargs["agg_map"]):
            if len(mapping) > 1:
                # Replace categorical with uniformly random draw of the
                # other potential categories.

                unique_vals = np.unique(X_LE[:, idx])
                ps = np.apply_along_axis(
                    lambda x, y: np.setdiff1d(y, x),
                    1,
                    X_obs_LE[:, idx, np.newaxis],
                    unique_vals,
                )
                replacements = np.apply_along_axis(np.random.choice, 1, ps)
                modified_max_distance[:, idx] = replacements
            else:
                modified_max_distance[:, idx] = np.where(
                    X_obs_LE[:, idx] < midpoint[:, idx],
                    max_value[:, idx],
                    min_value[:, idx],
                )
        return modified_max_distance

    max_value = np.tile(X.max(axis=0), (len(X_obs), 1))
    min_value = np.tile(X.min(axis=0), (len(X_obs), 1))
    midpoint = (min_value + max_value) / 2
    return np.where(X_obs < midpoint, max_value, min_value)


def gaussian(X: np.ndarray, sigma: int, **kwargs) -> np.ndarray:
    """Gaussian noise distribution

    Args:
        X (np.ndarray): source data
        sigma (int): sd of noise

    Returns:
        np.array: noisy source data
    """

    if categorical_perturbation_case(**kwargs):
        raise NotImplementedError(
            "Gaussian perturbation has not been implemented for categorical feature types."
        )

    return np.clip(randn(*X.shape) * sigma + X, a_min=X.min(), a_max=X.max())


def gaussian_blur(X: np.ndarray, sigma: int, **kwargs) -> np.ndarray:
    """Generate gaussian blur over features

    Args:
        X (np.ndarray): source data for guassian blur
        sigma (int): Gaussian filter sigma

    Returns:
        np.ndarray: blurred data
    """

    if categorical_perturbation_case(**kwargs):
        raise NotImplementedError(
            "Gaussian Blur perturbation has not been implemented for categorical feature types."
        )

    return gaussian_filter(X, sigma=sigma)


def marginal(X: np.ndarray, X_obs: np.ndarray, **kwargs) -> np.ndarray:

    """
    Sample over marginal distribution

    Args:
        X (np.ndarray): Source data for marginals
        X_obs (np.ndarray): data observations for which to sample

    Returns:
        np.ndarray: marginal sample
    """

    if categorical_perturbation_case(**kwargs):
        X_LE = ohe_to_le(X, kwargs["agg_map"])
        X_obs_LE = ohe_to_le(X_obs, kwargs["agg_map"])

        # Uniformly sample the marginals/features
        idx = np.random.randint(len(X_LE), size=X_obs_LE.shape)
        ret_mat = X_LE[idx, np.arange(X_obs_LE.shape[1])]
        return ret_mat

    idx = np.random.randint(len(X), size=X_obs.shape)
    return X[idx, np.arange(X_obs.shape[1])]


def gaussian_blur_permutation(
    X: np.ndarray, sigma: int, iterations=1000, **kwargs
) -> np.ndarray:
    """Gaussian blur over permuted features

    Args:
        X (np.ndarray): Source data for guassian blur
        sigma (int): Gaussian filter sigma
        iterations (int, optional): Number of permutations to average over.
                                    Defaults to 1000.

    Returns:
        np.ndarray: blurred data
    """

    shuffled_gaussian_X = np.zeros_like(X).astype(float)
    d = X.shape[1]

    # Generate unique permutations of features
    permutations = []
    perms = set()
    for _ in range(iterations):
        while True:
            perm = permutation(d)
            key = tuple(perm)
            if key not in perms:
                perms.update(key)
                permutations.append(perm)
                break

    # Average gaussian blur over permutations
    for p in permutations:
        shuffled_gaussian_X += gaussian_filter(X[:, p], sigma)

    shuffled_gaussian_X /= iterations

    return shuffled_gaussian_X


def training(X: np.ndarray, **kwargs) -> np.ndarray:
    """Training data distribution

    Args:
        X (np.ndarray): training data

    Returns:
        np.array: train dataset
    """

    if categorical_perturbation_case(**kwargs):
        X_LE = ohe_to_le(X, kwargs["agg_map"])
        return X_LE

    return X


def opposite_class(
    X: np.ndarray,
    y: np.ndarray,
    pred_y_obs: np.ndarray,
    nsamples: int,
    **kwargs,
) -> np.ndarray:
    """Samples with an opposite label from the observation prediction

    Args:
        X (np.ndarray): training data
        y (np.ndarray): class of training data
        pred_y_obs (np.ndarray): predicted class of observations
        nsamples (int): Number of samples
    Returns:
        np.array: sample of training data with opposite class
    """
    assert (
        nsamples is not None
    ), "nsamples must be specified for opposite_class"

    if categorical_perturbation_case(**kwargs):
        X_LE = ohe_to_le(X, kwargs["agg_map"])

        class_dict = {
            y_: sample(X_LE[y != y_], nsamples, random_state=None)
            for y_ in np.unique(y)
        }

        sample_sizes = [len(s) for s in class_dict.values()]
        assert all(s == min(sample_sizes) for s in sample_sizes), (
            f"Opposite class can only support a maximum sample size of {min(sample_sizes)}. "
            "The sample size is constrained by the smallest class in the training data. "
            "Please either use more data or decrease nsamples for opposite_class."
        )

        return np.array([class_dict[y_] for y_ in pred_y_obs])

    class_dict = {
        y_: sample(X[y != y_], nsamples, random_state=None)
        for y_ in np.unique(y)
    }

    sample_sizes = [len(s) for s in class_dict.values()]
    assert all(s == min(sample_sizes) for s in sample_sizes), (
        f"Opposite class can only support a maximum sample size of {min(sample_sizes)}. "
        "The sample size is constrained by the smallest class in the training data. "
        "Please either use more data or decrease nsamples for opposite_class."
    )

    return np.array([class_dict[y_] for y_ in pred_y_obs])


def nearest_neighbors(
    X: np.ndarray, X_obs: np.ndarray, k: int, **kwargs
) -> np.ndarray:
    """Nearest neighbors from reference set

    Args:
        X (np.ndarray): training data
        X_obs (np.ndarray): data observations for which to generate neighbors
        k (int): number of neighbors

    Returns:
        np.array: nearest neighbors
    """

    if categorical_perturbation_case(**kwargs):
        X_LE = ohe_to_le(X, kwargs["agg_map"])
        X_obs_LE = ohe_to_le(X_obs, kwargs["agg_map"])
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X_LE)
        nn = nbrs.kneighbors(X_obs_LE, return_distance=False)
        return X_LE[nn]

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X)
    nn = nbrs.kneighbors(X_obs, return_distance=False)
    return X[nn]


def nearest_neighbors_counterfactual(
    X: np.ndarray,
    y: np.ndarray,
    X_obs: np.ndarray,
    pred_y_obs: np.ndarray,
    k: int,
    **kwargs,
) -> np.ndarray:
    """Nearest neighbors from reference set having opposite class

    Args:
        X (np.ndarray): training data
        y (np.ndarray): class of training data
        X_obs (np.ndarray): data observations for which to generate neighbors
        pred_y_obs (np.ndarray): predicted class of observations
        k (int): number of neighbors

    Returns:
        np.array: nearest neighbors
    """
    classes = np.unique(y)

    if categorical_perturbation_case(**kwargs):
        X_LE = ohe_to_le(X, kwargs["agg_map"])
        X_obs_LE = ohe_to_le(X_obs, kwargs["agg_map"])

        # Create NN for each class containing other classes
        class_dict = {
            y_: NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
                X_LE[y != y_]
            )
            for y_ in classes
        }
        # Get indices of nn for each observation based on predicted class
        nn = np.concatenate(
            [
                class_dict[y_].kneighbors(
                    x.reshape(1, -1), return_distance=False
                )
                for (x, y_) in zip(X_obs_LE, pred_y_obs)
            ],
            axis=0,
        )

        return X_LE[nn]

    # Create NN for each class containing other classes
    class_dict = {
        y_: NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
            X[y != y_]
        )
        for y_ in classes
    }
    # Get indices of nn for each observation based on predicted class
    nn = np.concatenate(
        [
            class_dict[y_].kneighbors(x.reshape(1, -1), return_distance=False)
            for (x, y_) in zip(X_obs, pred_y_obs)
        ],
        axis=0,
    )

    return X[nn]
