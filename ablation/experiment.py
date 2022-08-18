import logging
import os
import pickle
import shutil
import time
from copy import copy
from typing import Any, Dict, List, Union
from warnings import warn

import numpy as np
import pandas as pd
import yaml

from .ablation import Ablation
from .baseline import generate_baseline_distribution
from .dataset import load_data
from .explanations import Explanations
from .perturb import generate_perturbation_distribution
from .pytorch_explanations import captum_explanation
from .pytorch_model import load_model, train
from .utils.evaluate import eval_model_performance
from .utils.logging import logger
from .utils.model import _predict_fn
from .utils.transform import le_to_ohe, ohe_to_le


class Config:
    def __init__(
        self,
        dataset_name: str,
        model_type: str,
        perturbation_config: Dict[str, Dict[str, int]],
        baseline_config: Dict[str, Dict[str, int]],
        explanation_methods: List[str],
        ablation_args: Dict[str, Any],
        dataset_sample_perc: float = 1.0,
        dataset_n_random_features: int = 0,
        n_trials: int = 1,
        path: Union[str, os.PathLike] = "tmp",
        load=False,
        rerun_ablation=False,
    ):
        """Experiment config

        Args:
            dataset_name (str): name of dataset
            model_type (str): model type ('nn' or 'linear')
            perturbation_config (Dict[str, Dict[str, int]]): dict of perturbation names associated with dicts of extra arguments
            baseline_config (Dict[str, Dict[str, int]]): dict of baseline names associated with dicts of extra arguments
            explanation_methods (List[str]): list of explanation methods
            ablation_args (Dict[str, Any]): dict of ablation arguments
            dataset_sample_perc (float): percent of dataset to use for experiment
            dataset_n_random_features (int): number of random features to add to dataset for sanity check
            n_trials (int): Number of seeds to run experiment
            path (str, optional): path to save results, model, and intermediary calculations. Defaults to "tmp".
            load (bool, optional): If true, will load from path. Defaults to False.
            rerun_ablation (bool, optional): If true, will rerun ablation on load
        """
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.dataset_sample_perc = dataset_sample_perc
        self.dataset_n_random_features = dataset_n_random_features
        self.perturbation_config = perturbation_config
        self.baseline_config = baseline_config
        self.explanation_methods = explanation_methods
        self.ablation_args = ablation_args
        self.n_trials = n_trials
        self.path = path
        self.load = load
        self.rerun_ablation = rerun_ablation

        self._check()

    def _check(self):

        assert (
            isinstance(self.n_trials, int) and self.n_trials >= 1
        ), "n_trials must be an integer >=1"

        assert (
            len(self.perturbation_config) > 0
        ), "Must specify at least one perturbation"

        assert (
            len(self.explanation_methods) > 0
        ), "Must specify at least one explanation method"

        assert len(self.baseline_config) > 0, "Must specify at least one baseline"

        assert self.model_type in [
            "nn",
            "linear",
        ], "model_type must be type 'nn' or 'linear'"

    @classmethod
    def from_yaml_file(cls, path: Union[str, os.PathLike]):
        config = yaml.safe_load(open(path, "r"))
        return cls(**config)

    @classmethod
    def from_dict(cls, dict_):
        return cls(**dict_)

    def to_dict(self):
        return copy(self.__dict__)

    def save(self):
        return yaml.dump(
            self.__dict__,
            open(os.path.join(self.path, "config.yml"), "w"),
            allow_unicode=True,
        )

    @classmethod
    def load(cls, path: Union[str, os.PathLike]):
        return cls.from_yaml_file(os.path.join(path, "config.yml"))

    @property
    def result_name(self):
        """Name of results file based on ablation args"""
        explanation = "local" if self.ablation_args["local"] else "global"
        return f"results-{explanation}"

    def diff(self, other: object) -> bool:
        experiment_args = [
            "dataset_name",
            "perturbation_config",
            "baseline_config",
            "explanation_methods",
            "ablation_args",
            "dataset_sample_perc",
            "dataset_n_random_features",
            "n_trials",
            "model_type",
        ]
        _self = self.__dict__
        _other = other.__dict__
        diff = [arg for arg in experiment_args if _self[arg] != _other[arg]]

        return diff

    def __eq__(self, other: object) -> bool:
        return len(self.diff(other)) == 0


class Experiment:
    def __init__(self, config: Config):
        """Experiment runner

        Args:
            config (Config): configuration
        """
        self.config = config

        self._clean_dir()
        self._config_check()
        self.config.save()
        self._set_logging()

        self.dataset = load_data(
            self.config.dataset_name,
            self.config.dataset_sample_perc,
            self.config.dataset_n_random_features,
        )

        if self.config.load:
            self.model = load_model(self.config.path)
            self.label_shuffed_model = load_model(
                self.config.path, prefix="label_shuffed_model"
            )
        else:
            self.model = train(
                self.dataset,
                max_epochs=100,
                path=self.config.path,
                model_type=self.config.model_type,
            )
            self.label_shuffed_model = train(
                self.dataset,
                max_epochs=100,
                path=self.config.path,
                model_type=self.config.model_type,
                prefix="label_shuffed_model",
                shuffle_labels=True,
            )

    def _set_logging(self):
        fh = logging.FileHandler(os.path.join(self.config.path, "experiment.log"))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def _config_check(self):
        self.rerun_ablation = self.config.rerun_ablation

        if self.config.load:
            if not os.path.exists(self.config.path):
                raise IOError(f"Path does not exist: {self.config.path}")

            old_config = Config.load(self.config.path)
            diff = self.config.diff(old_config)

            if diff == ["ablation_args"]:
                warn(
                    "Recomputing ablations from previous experiment with settings: "
                    f"{', '.join([f'{k}: {v}' for k,v in self.config.ablation_args.items()])}."
                )
                self.rerun_ablation = True

            elif len(diff) > 0:
                # TODO: Add more functionality to rerun components of the experiment
                raise ValueError(
                    f"Configuration file doesn't match. The following arguments have changed: {', '.join(diff)}."
                    # f"Using config.yml from {self.config.path}."
                    # "Set load = False to run new config."
                )

    def _clean_dir(self):
        """Clean experiment directory"""
        if os.path.exists(self.config.path) and not self.config.load:
            shutil.rmtree(self.config.path, ignore_errors=True)

        if not os.path.exists(self.config.path):
            os.makedirs(self.config.path)

    def _load_if_exists(self, name):
        """Load file if exists"""
        file = os.path.join(self.config.path, f"{name}.pkl")
        if os.path.exists(file):
            return pickle.load(open(file, "rb"))
        return

    def _save(self, obj, name):
        """Save object as pkl file"""
        file = os.path.join(self.config.path, f"{name}.pkl")
        pickle.dump(obj, open(file, "wb"))

    def _compute_model_sanity_checks(self):
        self.model_sanity_checks = self._load_if_exists("model_sanity_checks")

        if self.model_sanity_checks is None:
            self.model_sanity_checks = {}

            self.model_sanity_checks["label_shuffled"] = {
                scoring_method: eval_model_performance(
                    self.label_shuffed_model,
                    self.dataset.X_test,
                    self.dataset.y_test,
                    scoring_method=scoring_method,
                )
                for scoring_method in self.config.ablation_args["scoring_methods"]
            }

            self._save(self.model_sanity_checks, "model_sanity_checks")

        # TODO: Do we want also the default probabilities?
        # _, counts = np.unique(self.dataset.y_train, return_counts=True)
        # naive_pred = lambda x: np.tile(counts / sum(counts), (len(x), 1))
        # self.naive_performance = eval_model_performance(
        #     naive_pred,
        #     self.dataset.X_test,
        #     self.dataset.y_test,
        #     scoring_method=config.ablation_args["scoring_method"],
        # )

    def _compute_perturbations(
        self, perturbation_config: Dict[str, Dict[str, int]]
    ) -> None:
        """Compute perturbations

        Args:
            perturbation_config (Dict[str, Dict[str, int]]): dict of perturbation names associated with dicts of extra arguments
        """
        self.perturbations = self._load_if_exists("perturbations")
        if self.perturbations is None:
            self.perturbations = {
                trial: {
                    name: generate_perturbation_distribution(
                        name,
                        self.dataset.X_train,
                        self.dataset.X_test,
                        random_state=trial,
                        agg_map=self.dataset.agg_map,
                        **kwargs,
                    )
                    for name, kwargs in perturbation_config.items()
                }
                for trial in range(self.config.n_trials)
            }
            self._save(self.perturbations, "perturbations")

    def _compute_baselines(self, baseline_config: Dict[str, Dict[str, int]]) -> None:
        """Compute baselines

        Args:
            baseline_config (Dict[str, Dict[str, int]]): dict of baseline names associated with dicts of extra arguments
        """

        self.baselines = self._load_if_exists("baselines")

        if self.baselines is None:
            numpy_model = _predict_fn(self.model)
            y = numpy_model(self.dataset.X_train)
            y_obs = numpy_model(self.dataset.X_test)

            self.baselines = {
                trial: {
                    name: generate_baseline_distribution(
                        name,
                        self.dataset.X_train,
                        self.dataset.X_test,
                        y,
                        y_obs,
                        random_state=trial,
                        agg_map=self.dataset.agg_map,
                        **kwargs,
                    )
                    for name, kwargs in baseline_config.items()
                }
                for trial in range(self.config.n_trials)
            }
            self._save(self.baselines, "baselines")

    def _compute_explanations(
        self,
        explanation_methods: List[str],
        computed_baselines: Dict[str, np.ndarray],
    ) -> None:
        """Compute explanations

        Args:
            explanation_methods (List[str]): list of explanations methods
            computed_baselines (Dict[str, np.ndarray]): dict of baseline name with associated baselines
        """

        # TODO, we want to support raw explanations in addition to Explanations object
        self.explanations = self._load_if_exists("explanations")

        if self.explanations is None:
            self.explanations = {}
            non_random_exp_methods = [m for m in explanation_methods if m != "random"]
            for trial, baselines in computed_baselines.items():
                self.explanations[trial] = {}
                for method in non_random_exp_methods:
                    self.explanations[trial][method] = {}
                    for bname, baseline in baselines.items():
                        logger.info(
                            f"Running explanation: {method} | baseline: {bname}"
                        )

                        # Calculate the explanation values for set of observations
                        explanations = Explanations(
                            explanation_values=captum_explanation(
                                method,
                                self.model,
                                self.dataset.X_test,
                                baseline,
                                random_state=trial,
                            ),
                            agg_map=self.dataset.agg_map,
                        )

                        self.explanations[trial][method][bname] = explanations

                    if "random" in explanation_methods:
                        logger.info("Running random explanation")
                        explanations = Explanations(
                            explanation_values=captum_explanation(
                                "random",
                                self.model,
                                self.dataset.X_test,
                                baseline,
                                random_state=trial,
                            ),
                            agg_map=self.dataset.agg_map,
                        )
                        self.explanations[trial][method][
                            "random explanation"
                        ] = explanations

            self._save(self.explanations, "explanations")

    def run_exp(self):
        """Run experiment"""

        self._compute_model_sanity_checks()
        self._compute_perturbations(self.config.perturbation_config)
        self._compute_baselines(self.config.baseline_config)
        self._compute_explanations(self.config.explanation_methods, self.baselines)

        self.results = self._load_if_exists(self.config.result_name)

        if self.results is None or self.rerun_ablation:

            trials = []
            for trial in range(self.config.n_trials):
                logger.info(f"Running ablation trial {trial}")
                np.random.seed(trial)
                comb = []
                for p_name, perturb in self.perturbations[trial].items():
                    for exp_name, exp_dict in self.explanations[trial].items():
                        for b_name, exp in exp_dict.items():

                            # TODO: debug kernelshap and lime for overflow
                            # Below is currently a quick fix to exclude these samples
                            overflow_idx = np.unique(
                                np.where(exp.data("sparse") > 10)[0]
                            )
                            if len(overflow_idx) > 0:
                                logger.warn(
                                    f"Overflow warning (perturb: {p_name}, exp: {exp_name}, baseline: {b_name}):\n"
                                    f"{len(overflow_idx)} samples removed. "
                                    f"Indices: {','.join(overflow_idx.astype(str))}"
                                )

                            abl = Ablation(
                                perturbation_distribution=np.delete(
                                    perturb, overflow_idx, 0
                                ),
                                model=self.model,
                                dataset=self.dataset,
                                X=np.delete(self.dataset.X_test, overflow_idx, 0),
                                y=np.delete(self.dataset.y_test, overflow_idx, 0),
                                explanation_values=np.delete(
                                    exp.data("sparse"), overflow_idx, 0
                                ),
                                explanation_values_dense=np.delete(
                                    exp.data("dense"), overflow_idx, 0
                                ),
                                random_feat_idx=self.dataset.dense_random_feat_idx,
                                **self.config.ablation_args,
                            )
                            # abl = Ablation(
                            #     perturbation_distribution=perturb,
                            #     model=self.model,
                            #     X=self.dataset.X_test,
                            #     y=self.dataset.y_test,
                            #     explanation_values=exp,
                            #     random_feat_idx=self.dataset.random_feat_idx,
                            #     **self.config.ablation_args,
                            # )

                            result = abl.ablate_features()

                            n_obs = len(result["scores"])
                            result["explanation_method"] = [exp_name] * n_obs
                            result["baseline"] = [b_name] * n_obs
                            result["perturbation"] = [p_name] * n_obs
                            (
                                result["random_sanity_check_idx"],
                                result["random_sanity_check_perc"],
                            ) = abl.random_sanity_check_idx()
                            result[
                                "random_sanity_check_value"
                            ] = abl.random_sanity_check_value()
                            comb.append(result)

                trial_df = pd.concat(comb)
                trial_df["trial"] = trial
                trials.append(trial_df)

            self.results = pd.concat([t for t in trials]).reset_index(drop=True)
            self._save(self.results, self.config.result_name)

        return self.results
