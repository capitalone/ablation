"""
ablation.py
About: Load ablation methods for Ablation
"""
from copy import deepcopy
from itertools import chain
from typing import Dict, List

import numpy as np
import pandas as pd

from .dataset import NumpyDataset
from .utils.evaluate import append_dict_lists, eval_model_performance
from .utils.transform import le_to_ohe, ohe_to_le


def local_explanation_importance(exp: np.ndarray, relative=False) -> np.ndarray:
    """Average absolute value of ranked local explanations. For example, the first
        value in the returned array is the average of largest local absolute feature
        importance across all samples.

    Args:
        exp (np.ndarray): local explanations of shape (samples, features)
        relative (bool): return importance relative to sum of all features

    Returns:
        np.ndarray: average importance of ranked local explanations
    """
    abs_exp = np.abs(exp)
    if relative:
        abs_exp = abs_exp / abs_exp.sum(-1, keepdims=True)
    ordered_exp = np.sort(abs_exp, -1)[:, ::-1]
    avg_importance = ordered_exp.mean(0)
    return avg_importance


class Ablation:
    """Class to perform Ablations using feature importance"""

    def __init__(
        self,
        perturbation_distribution: np.ndarray,
        model,
        dataset: NumpyDataset,
        X: np.ndarray,
        y: np.ndarray,
        explanation_values: np.ndarray,
        explanation_values_dense: np.ndarray,
        random_feat_idx: np.ndarray = None,
        local: bool = True,
        scoring_methods: List[str] = ["log_loss", "abs_diff", "auroc"],
        scoring_step: float = 0,
    ):
        """Constructor for ablation

        Args:
            perturbation_distribution (np.ndarray): perturbed dataset as a numpy arr
            model ([type]): a pre-trained model
            X (np.ndarray): original features dataset
            y (np.ndarray): target values
            explanation_values (np.ndarray): explanation values used for feature importance (eg. shap values)
            random_feat_idx (np.ndarray): random feature index
            local (bool): If True, perturbs data based on rank ordered local rather than global explanations.
                          Defaults to True.
            scoring_methods (List[str]): List of scoring methods ('log_loss','abs_diff', 'auroc', 'aupr')
            scoring_step (float): Fraction of features to ablate in-between model evaluations (1, 0]. If 0, will
                                  evaluate at every feature.

        """

        self.perturbation_distribution = perturbation_distribution
        self.model = model
        self.dataset = deepcopy(dataset)
        self.X = deepcopy(X)
        self.y = deepcopy(y)
        self.explanation_values = explanation_values
        self.explanation_values_dense = explanation_values_dense
        self.random_feat_idx = random_feat_idx
        self.local = local

        for s in scoring_methods:
            assert s in [
                "auroc",
                "aupr",
                "log_loss",
                "abs_diff",
            ], f"{s} is not a valid scoring method! "
        self.scoring_methods = scoring_methods

        assert (
            scoring_step < 1 and scoring_step >= 0
        ), "scoring step must be between 0 and 1"

        self.scoring_step = scoring_step

    def _steps(self):
        """Calculate steps at which to evaluate model performance
            1. perturb dataset at atleast every feature
            2. record model evaluation at each score step
            3. provide normalized pct step for each evaluation point

        Returns:
            Tuple[np.ndarray,np.ndarray,np.ndarray]: perturb steps, score steps, percent steps
        """

        # n_features = self.X.shape[1]
        n_features = len(self.dataset.original_feature_names)
        # fraction of features included at each step
        if self.scoring_step == 0:
            # If scoring_step is 0 default to scoring at each feature
            pct_steps = np.arange(0, n_features + 1) / n_features
        else:
            pct_steps = np.arange(0, 1 + self.scoring_step, self.scoring_step)
        # number of features at which to provide evaluation (exclude initial evaluation with all features)
        score_steps = (pct_steps * (n_features - 1)).astype(int)[1:]

        # Number of perturbations at minimum = n_features
        # Additional repeat steps taken if number of scoring steps> n_features
        perturb_steps = (
            score_steps if len(score_steps) > n_features else np.arange(n_features)
        )

        return perturb_steps, score_steps, pct_steps

    def ablate_features(self, plot=False) -> Dict[str, np.ndarray]:
        """Main function for ablation

        Args:
            plot (bool, optional): Plot ablation. Defaults to False.

        Returns:
            np.ndarray: scores, normalized scores, and percent of feature steps
        """
        perturb_steps, score_steps, pct_steps = self._steps()
        ranked_features = self._sorted_feature_rankings()

        # Convert dataset to LE from OHE
        self.X = ohe_to_le(self.X, self.dataset.agg_map)

        scores = {s: [] for s in self.scoring_methods}
        score_no_perturbation = self._get_model_performance()
        scores = append_dict_lists(scores, score_no_perturbation)

        indx = np.arange(len(self.X))
        # Replace features in order of importance with perturbation, calculate and store new score
        for i in perturb_steps:
            # Replace at feature_indx (i-th highest ) with perturbation distribution
            feature_idx = ranked_features[i]
            self.X[indx, feature_idx] = self.perturbation_distribution[
                indx, feature_idx
            ]
            if i in score_steps:
                score_perturbed = self._get_model_performance()
                scores = append_dict_lists(scores, score_perturbed)

        n_steps = len(pct_steps)
        n_scores = len(self.scoring_methods)

        scores = {k: np.array(s) for k, s in scores.items()}
        score_change = {
            k: np.concatenate([s[1:] - s[:-1], [0]]) for k, s in scores.items()
        }
        exp_importance = np.concatenate(
            [local_explanation_importance(self.explanation_values_dense), [0]]
        )

        results = {
            "pct_steps": list(pct_steps) * n_scores,
            "score_name": sum([[k] * n_steps for k in scores], []),
            "scores": np.concatenate(list(scores.values())),
            "score_change": np.concatenate(list(score_change.values())),
            "exp_importance": list(exp_importance) * n_scores,
        }

        # Restore Data Format
        # NOTE Test if self.X here equals self.dataset.X
        # self.X = le_to_ohe(self.X,self.dataset.agg_map)

        return pd.DataFrame(results)

    def _get_model_performance(self) -> np.ndarray:
        """Return performance for a model's paredicted probabilities

        Returns:
            float: model performance
        """
        return {
            s: eval_model_performance(
                self.model, le_to_ohe(self.X, self.dataset.agg_map), self.y, s
            )
            for s in self.scoring_methods
        }

    def _sorted_feature_rankings(self):
        """Generate ranked feature list, given explanation values

        Returns:
            np.array: Array of feature indices, in order of feature importance
                     If local, will return with shape (features, samples)
        """

        # vals = np.abs(self.explanation_values)
        vals = np.abs(self.explanation_values_dense)
        if not self.local:
            vals = vals.mean(0)

        return np.argsort(-vals).transpose()

    def random_sanity_check_idx(self) -> float:
        """Sanity check for random features

        Global: first index where a random feature appears in ranked global feature importance
        Local: minimum weighted rank among all random features

        Returns:
            float: random feature rank, random feature percent rank
        """

        if self.random_feat_idx is None:
            return self.X.shape[1], 1.0

        ranked_features = self._sorted_feature_rankings()
        if self.local:

            min_rank = len(ranked_features)
            for rand_feat_idx in self.random_feat_idx:
                is_random_feat = ranked_features == rand_feat_idx
                weighted_random_rank = sum(
                    is_random_feat.sum(1)
                    * np.arange(len(is_random_feat))
                    / is_random_feat.sum()
                )
                min_rank = min(min_rank, weighted_random_rank)

            # median_rank = np.where(
            #     is_random_feat.sum(1).cumsum() > is_random_feat.sum() / 2
            # )[0].min()

            return min_rank, min_rank / self.X.shape[1]

        is_random_feat = np.in1d(ranked_features, self.random_feat_idx)
        random_rank = np.where(is_random_feat)[0].min()

        return random_rank, random_rank / self.X.shape[1]

    def random_sanity_check_value(self) -> float:
        """Sanity check for random feature explanation values

        Returns:
            float: Maximum value of most important random feature global explanation
        """

        if self.random_feat_idx is None:
            return 0

        global_importance = np.abs(self.explanation_values).mean(0)
        random_feat_importance = global_importance[self.random_feat_idx].max()

        return random_feat_importance
