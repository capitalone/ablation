from typing import List

import numpy as np


def ohe_to_le(X: np.ndarray, agg_map: List[List[int]]) -> np.ndarray:
    """
    Mixed data with One-Hot-Encoded categoricals
    to Mixed data with Label Encoded categoricals.

    An example of the aggregation mappings are:

    agg_map = [[1,2,3],[4],[5]]

    In this example, the dense feature index 0 maps to
    sparse feature indices 1,2,3.  Dense feature index 0 is likely
    a categorical that was split into three features via
    one-hot-encoding.  Dense feature indices 1 and 2 are one-to-one
    mappings, meaning that they represent non-categoricals.

    Args:
        X (np.ndarray): Mixed data to be transformed
        agg_map (list[list[int]]): Aggregation map

    Returns:
        np.ndarray: Mixed data with label encoded categoricals
    """
    if agg_map is None:
        return X

    X_LE = np.zeros((X.shape[0], len(agg_map)))
    for idx, mapping in enumerate(agg_map):
        # Look for multiple mappings due to categorical OHE.
        if len(mapping) > 1:
            X_LE[:, idx] = np.argmax(X[:, mapping], axis=1)
        else:
            X_LE[:, idx] = X[:, mapping[0]]

    return X_LE


def le_to_ohe(X: np.ndarray, agg_map: List[List[int]]) -> np.ndarray:
    """
    Transforms mixed data with Label Encoded categoricals
    to mixed data with One-Hot-Encoded categoricals.

    Args:
        X (np.ndarray): Mixed data with label encoded categoricals
        agg_map (list[list[int]]): Aggregation map

    Returns:
        np.ndarray: Mixed data with one-hot-encoded categoricals
    """

    # If agg_map not provided, return input.
    if agg_map is None:
        return X

    X_OHE = np.zeros((X.shape[0], sum(len(mapping) for mapping in agg_map)))
    for idx, mapping in enumerate(agg_map):

        # If categorical mapping
        if len(mapping) > 1:
            # NOTE: Inputs are rounded during LE-to-OHE transformation.
            X_OHE[:, mapping] = np.eye(len(mapping))[
                np.round(X[:, idx]).astype(int)
            ]

        # else, numerical mapping
        else:
            X_OHE[:, mapping[0]] = X[:, idx]

    return X_OHE
