import numpy as np

from ablation.dataset import load_data


def test_dataset_sampling():
    data_full = load_data("synthetic")
    data_sampled = load_data("synthetic", 0.5)
    _, full_count = np.unique(data_full.y_train, return_counts=True)
    _, sampled_count = np.unique(data_sampled.y_train, return_counts=True)
    half_full = full_count[0] // 2
    assert half_full == sampled_count[0] or half_full + 1 == sampled_count[0]


def test_dataset_random_features():
    data = load_data("synthetic", n_rand_features=0)
    data_rand = load_data("synthetic", n_rand_features=1)
    assert data.X_train.shape[1] + 1 == data_rand.X_train.shape[1]
