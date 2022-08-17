import logging

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ablation.dataset import NumpyDataset, load_data
from ablation.pytorch_model import train

logging.basicConfig(format="%(message)s", level=logging.WARNING)

TEST_RANDOM_STATE = 42


@pytest.fixture(scope="session")
def binary_dataset():
    n_features = 10
    feature_names = [f"FEATURE{idx}" for idx in range(n_features)]
    X, y = make_classification(
        n_samples=100,
        n_features=n_features,
        n_informative=5,
        n_classes=2,
        random_state=TEST_RANDOM_STATE,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=TEST_RANDOM_STATE
    )

    return NumpyDataset(
        X_train,
        y_train,
        X_test,
        y_test,
        n_classes=2,
        feature_names=feature_names,
        original_feature_names=feature_names,
    )


@pytest.fixture(scope="session")
def multiclass_dataset():
    n_features = 10
    feature_names = [f"FEATURE{idx}" for idx in range(n_features)]
    X, y = make_classification(
        n_samples=100,
        n_features=n_features,
        n_informative=5,
        n_classes=4,
        random_state=TEST_RANDOM_STATE,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=TEST_RANDOM_STATE
    )

    return NumpyDataset(
        X_train,
        y_train,
        X_test,
        y_test,
        n_classes=4,
        feature_names=feature_names,
        original_feature_names=feature_names,
    )


@pytest.fixture(scope="session")
def small_dataset():
    X_test = np.array([[1, 2, 3], [1, 2, 4], [2, 1, 4]])
    y_test = np.array([0, 1, 1])
    return X_test, y_test


@pytest.fixture(scope="session")
def small_numpydataset(small_dataset):
    (X_test, y_test) = small_dataset
    return NumpyDataset(
        X_train=X_test,
        y_train=y_test,
        X_test=X_test,
        y_test=y_test,
        n_classes=2,
        feature_names=["Feature1", "Feature2", "Feature3"],
        original_feature_names=["Feature1", "Feature2", "Feature3"],
    )


@pytest.fixture(scope="session")
def small_model():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _model(x):
        return sigmoid(x @ np.array([5, 3, 1]) - 15)

    return _model


@pytest.fixture(scope="session")
def small_explanations():
    return np.array([[5, 6, 3], [5, 6, 4], [10, 3, 4]])


@pytest.fixture(scope="session")
def local_rank_small_explantions():
    return np.array([[1, 0, 2], [1, 0, 2], [0, 2, 1]])


@pytest.fixture(scope="session")
def global_rank_small_explantions():
    return np.array([0, 1, 2])


@pytest.fixture(scope="session")
def small_perturbation():
    return np.zeros((3, 3))


@pytest.fixture(scope="session")
def synthetic_model():
    data = load_data("synthetic")
    return train(data, max_epochs=1, path="./test_logs")
