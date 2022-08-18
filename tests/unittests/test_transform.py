# pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb

import numpy as np

# from numpy import testing
import pytest
from numpy import testing

from ablation.dataset import load_data
from ablation.utils.transform import le_to_ohe, ohe_to_le


def test_categorical_to_le_simple_001():
    n_samples = 100
    X_OHE = np.tile([0, 1, 0, 3, 4, -5], [n_samples, 1])
    agg_map = [[0, 1, 2], [3], [4], [5]]
    solution = np.tile([1, 3, 4, -5], [n_samples, 1])
    testing.assert_array_equal(ohe_to_le(X_OHE, agg_map), solution)


def test_categorical_to_le_simple_002():
    n_samples = 100
    X_OHE = np.tile([0, 0, 1, 3, 4, -5], [n_samples, 1])
    agg_map = [[0, 1, 2], [3], [4], [5]]
    solution = np.tile([2, 3, 4, -5], [n_samples, 1])
    testing.assert_array_equal(ohe_to_le(X_OHE, agg_map), solution)


def test_categorical_to_le_simple_003():
    n_samples = 100
    X_OHE = np.tile([0, 0, 1, 3, 1, 0], [n_samples, 1])
    agg_map = [[0, 1, 2], [3], [4, 5]]
    solution = np.tile([2, 3, 0], [n_samples, 1])
    testing.assert_array_equal(ohe_to_le(X_OHE, agg_map), solution)


def test_categorical_to_ohe_simple_001():
    n_samples = 100
    X_LE = np.tile([2, 3, 0], [n_samples, 1])
    agg_map = [[0, 1, 2], [3], [4, 5]]
    solution = np.tile([0, 0, 1, 3, 1, 0], [n_samples, 1])
    testing.assert_array_equal(le_to_ohe(X_LE, agg_map), solution)


def test_agg_map_format_handling_001():
    # Verify that error is thrown vs. failing silently
    # when incorrectly formatted agg_map is passed
    n_samples = 100
    X_LE = np.tile([2, 3, 0], [n_samples, 1])
    agg_map = [[0, 1, 2], 3, [4, 5]]  # Incorrect format
    with testing.assert_raises(TypeError):
        le_to_ohe(X_LE, agg_map)


def test_agg_map_format_handling_002():
    # Verify that error is thrown vs. failing silently
    # when incorrectly formatted agg_map is passed
    n_samples = 100
    X_OHE = np.tile([0, 0, 1, 3, 1, 0], [n_samples, 1])
    agg_map = [[0, 1, 2], 3, [4, 5]]  # Incorrect format
    with testing.assert_raises(TypeError):
        ohe_to_le(X_OHE, agg_map)


def test_transform_on_categorical():
    # Test dataset with many categoricals
    dataset = load_data("german")
    testing.assert_array_equal(
        le_to_ohe(ohe_to_le(dataset.X_test, dataset.agg_map), dataset.agg_map),
        dataset.X_test,
    )


def test_transform_on_numerical():
    # Test dataset with no categoricals, just numericals
    dataset = load_data("spambase")
    testing.assert_array_equal(
        le_to_ohe(ohe_to_le(dataset.X_test, dataset.agg_map), dataset.agg_map),
        dataset.X_test,
    )


def test_none_agg_map():
    # Test case when aggregation map is None.  Input should be returned.
    dataset = load_data("german")
    assert np.unique(ohe_to_le(dataset.X_test, None) == dataset.X_test) == True
