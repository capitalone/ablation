from typing import List

import numpy as np

from ablation.dataset import NumpyDataset


class Explanations:
    """
    Explanations is an object that allows for quick conversion/access
    to both sparse and dense explanations.  These explanation values
    depend on the composition of types used for the specific dataset
    under test.  It leverages the feature name mappings that are typically
    created by library transformer objects (e.g., ColumnTransformer from sklearn)
    """

    def __init__(self, explanation_values: np.ndarray, agg_map: List[List[int]]):
        """Constructor for Explanations class

        Args:
            explanation_values (np.ndarray): explanation values in sparse (i.e., post-encoded) format.
            agg_map (List[List[int]]): Aggregation map for sparse to dense.
        """

        self.explanation_values = explanation_values
        self.agg_map = agg_map

        # Calculate the dense explanation values
        self.dense_explanation_values = self.transform_explanations_sparse_to_dense(
            self.explanation_values
        )

    def data(self, format: str = "dense") -> np.ndarray:
        """
        Getter for the underlying np.ndarray structure that contains the explanation values.
        Those values are either sparse (not aggregated) or dense (aggregated).

        Args:
            format (str): "dense" or "sparse" formats to obtain np.ndarray explanations.

        Returns:
            np.array: explanation values
        """

        if format.lower() == "dense":
            return self.dense_explanation_values
        elif format.lower() == "sparse":
            return self.explanation_values
        else:
            raise ValueError(
                f"Supported formats for raw explanations are 'dense' and 'sparse'.  Got '{format.lower()}'."
            )

    def transform_explanations_sparse_to_dense(self, sparse_explanations: np.ndarray):
        """
        Calculates dense explanations from sparse explanations using class member aggregation map

        Args:
            sparse_explanations (np.ndarray): Explanations in sparse format.

        Returns:
            np.array: Explanations in dense format.
        """

        # Assume input is local sparse SHAP values
        sparse_rep_local_shap = sparse_explanations

        # If the aggregation map is None, indicating that there is no difference between
        # sparse and dense representations, simply return sparse_rep_local_shap.
        if self.agg_map is None:
            return sparse_rep_local_shap

        dense_rep_local_shap = np.zeros(
            (sparse_rep_local_shap.shape[0], len(self.agg_map))
        )
        for (i, map) in enumerate(self.agg_map):
            dense_rep_local_shap[:, i] = np.sum(sparse_rep_local_shap[:, map], axis=1)
        return dense_rep_local_shap
