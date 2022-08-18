import numpy as np
import pytest
from numpy import testing

from ablation.dataset import NumpyDataset
from ablation.explanations import Explanations


@pytest.fixture
def dataset():

    return NumpyDataset(
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        n_classes=1,
        feature_names=["f1", "f2_0", "f2_1", "f3", "f4_0", "f4_1"],
        original_feature_names=["f1", "f2", "f3", "f4"],
    )


def test_explanation_aggregation(dataset):
    exp_values = np.array([[1, 2, 3, 4, 5, 6], [-1, 1, -1, 0, 1, 0]])
    true_dense_values = np.array([[1, 5, 4, 11], [-1, 0, 0, 1]])
    exp = Explanations(exp_values, dataset.agg_map)

    testing.assert_array_equal(true_dense_values, exp.data("dense"))
