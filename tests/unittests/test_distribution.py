# pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb

import numpy as np
from numpy import testing

from ablation.baseline import generate_baseline_distribution
from ablation.dataset import load_data
from ablation.distributions import max_distance
from ablation.perturb import generate_perturbation_distribution
from ablation.utils.transform import le_to_ohe, ohe_to_le


def test_black_perturbation(binary_dataset):

    perturb = generate_perturbation_distribution(
        "constant", binary_dataset.X_train, binary_dataset.X_test, value=0
    )
    assert perturb.shape == binary_dataset.X_test.shape


def test_constant_value_perturbation(binary_dataset):

    perturb = generate_perturbation_distribution(
        "constant", binary_dataset.X_train, binary_dataset.X_test, value=1
    )
    assert perturb.mean() == 1


def test_perturbation_kwargs(binary_dataset):

    testing.assert_raises(
        AssertionError,
        testing.assert_array_equal,
        generate_perturbation_distribution(
            "gaussian", binary_dataset.X_train, binary_dataset.X_test, sigma=2,
        ),
        generate_perturbation_distribution(
            "gaussian", binary_dataset.X_train, binary_dataset.X_test, sigma=3,
        ),
    )


def test_perterbation_seed(binary_dataset):

    testing.assert_array_equal(
        generate_perturbation_distribution(
            "gaussian", binary_dataset.X_train, binary_dataset.X_test, sigma=2,
        ),
        generate_perturbation_distribution(
            "gaussian", binary_dataset.X_train, binary_dataset.X_test, sigma=2,
        ),
    )

    testing.assert_raises(
        AssertionError,
        testing.assert_array_equal,
        generate_perturbation_distribution(
            "gaussian",
            binary_dataset.X_train,
            binary_dataset.X_test,
            sigma=2,
            random_state=1,
        ),
        generate_perturbation_distribution(
            "gaussian",
            binary_dataset.X_train,
            binary_dataset.X_test,
            sigma=2,
            random_state=2,
        ),
    )

def test_max_distance():
    dataset = load_data("german")
    kwargs = {'agg_map' : dataset.agg_map}
    X_out = max_distance(X=dataset.X_train, X_obs=dataset.X_test, **kwargs)
    X_obs_LE_manual = ohe_to_le(dataset.X_test,dataset.agg_map)
    X_LE_manual = ohe_to_le(dataset.X_test,dataset.agg_map)

    # Check that no label encoded values match from X_test and X_train
    for idx, mapping in enumerate(dataset.agg_map):
        if(len(mapping)>1):
            # Verify unique set of values
            testing.assert_array_equal( np.unique(X_out[:,idx]),
                                        np.unique(X_LE_manual[:,idx]))

            # Verify that values are changed to something different
            # than their original value.
            assert(0 not in X_out[:,idx]-X_obs_LE_manual[:,idx])

def test_baseline_seed(binary_dataset):

    testing.assert_array_equal(
        generate_baseline_distribution(
            "gaussian",
            binary_dataset.X_train,
            binary_dataset.X_test,
            sigma=2,
            nsamples=10,
            random_state=1,
        ),
        generate_baseline_distribution(
            "gaussian",
            binary_dataset.X_train,
            binary_dataset.X_test,
            sigma=2,
            nsamples=10,
            random_state=1,
        ),
    )

    testing.assert_array_equal(
        generate_baseline_distribution(
            "opposite_class",
            X=binary_dataset.X_train,
            X_obs=binary_dataset.X_test,
            y=binary_dataset.y_train,
            y_obs=binary_dataset.y_test,
            nsamples=10,
            random_state=2,
        ),
        generate_baseline_distribution(
            "opposite_class",
            X=binary_dataset.X_train,
            X_obs=binary_dataset.X_test,
            y=binary_dataset.y_train,
            y_obs=binary_dataset.y_test,
            nsamples=10,
            random_state=2,
        ),
    )
