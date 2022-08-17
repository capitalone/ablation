# from ablation.perturb import X_test
import numpy as np
import pytest
from numpy import testing

from ablation.ablation import Ablation


@pytest.fixture
def ablation_instance(
    small_perturbation, small_model, small_numpydataset, small_explanations
):

    # NOTE This tests the numerical case where the small explanations
    # are the same for sparse and dense representation.
    return Ablation(
        perturbation_distribution=small_perturbation,
        model=small_model,
        dataset=small_numpydataset,
        X=small_numpydataset.X_test,
        y=small_numpydataset.y_test,
        explanation_values=small_explanations,
        explanation_values_dense=small_explanations,
        local=True,
    )


def test_global_ranking(ablation_instance, global_rank_small_explantions):

    ablation_instance.local = False
    testing.assert_array_equal(
        ablation_instance._sorted_feature_rankings(),
        global_rank_small_explantions,
    )


def test_local_ranking(ablation_instance, local_rank_small_explantions):

    testing.assert_array_equal(
        ablation_instance._sorted_feature_rankings(),
        local_rank_small_explantions.transpose(),
    )


def test_random_feature_sanity_check(ablation_instance):

    ablation_instance.random_feat_idx = np.array([0])
    local_rank, local_perc_rank = ablation_instance.random_sanity_check_idx()
    assert local_rank == 2 / 3

    ablation_instance.local = False
    global_rank, global_perc_rank = ablation_instance.random_sanity_check_idx()
    assert global_rank == 0


def test_random_feature_sanity_check2(ablation_instance):

    ablation_instance.random_feat_idx = np.array([1, 2])
    local_rank, local_perc_rank = ablation_instance.random_sanity_check_idx()
    assert local_rank == 2 / 3

    ablation_instance.local = False
    global_rank, global_perc_rank = ablation_instance.random_sanity_check_idx()
    assert global_rank == 1
