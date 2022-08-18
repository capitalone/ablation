import os
import pickle

import numpy as np
import pandas as pd

datasets = ["adult", "german", "har", "spambase"]
model = "nn"
attributions = ["local", "global"]
results_file = "results"

explanation_methods = ["deep_shap", "kernel_shap", "integrated_gradients"]
perturbations = ["constant_median", "marginal", "max_distance"]
baselines = [
    "nearest_neighbors",
    "opposite_class",
    "constant_median",
    "training",
]


def trap_area(x, y):
    areas = 0.5 * (x[1:] - x[:-1]) * (y[1:] + y[:-1])
    return areas.sum()


def area_between(x, y1, y2):
    """
    Area between y1 and y2
    If y1 is below y2, the area is negative, otherwise positive
    """
    return trap_area(x, y1) - trap_area(x, y2)


def area_under(x, y1, y2):
    """Area above y2 only under y1"""
    areas1 = 0.5 * (x[1:] - x[:-1]) * (y1[1:] + y1[:-1])
    areas2 = 0.5 * (x[1:] - x[:-1]) * (y2[1:] + y2[:-1])
    return -np.clip(areas2 - areas1, -1e10, 0).sum()


areas = []
for dataset in datasets:
    for attr in attributions:
        folder_path = f"../kdd_configs/kdd_experiments/{dataset}-{model}-{attr}/"
        model_sanity_check = pickle.load(
            open(os.path.join(folder_path, "model_sanity_checks.pkl"), "rb",)
        )["label_shuffled"]["auroc"]
        results_df = pd.read_pickle(os.path.join(folder_path, f"{results_file}-{attr}.pkl"))
        results_df["model_sanity_check"] = model_sanity_check
        for e in explanation_methods:
            for p in perturbations:
                for b in baselines:

                    random = results_df.query(
                        f"explanation_method=='{e}' and baseline == 'random explanation' and perturbation == '{p}' and score_name== 'auroc' "
                    )[["pct_steps", "scores", "trial"]].rename(
                        columns={"scores": "scores_random"}
                    )
                    other = results_df.query(
                        f"explanation_method=='{e}' and baseline == '{b}' and perturbation == '{p}' and score_name== 'auroc' "
                    )[
                        [
                            "pct_steps",
                            "scores",
                            "trial",
                            "model_sanity_check",
                            "random_sanity_check_perc",
                        ]
                    ]
                    test = random.merge(other)
                    area = (
                        test.groupby(["trial"])
                        .apply(
                            lambda x: area_between(
                                x["pct_steps"].values,
                                x["scores_random"].values,
                                x["scores"].values,
                            )
                        )
                        .mean()
                    )
                    area_above_curve_under_sanity = (
                        test.query("pct_steps<=random_sanity_check_perc")
                        .groupby(["trial"])
                        .apply(
                            lambda x: area_under(
                                x["pct_steps"].values,
                                x["model_sanity_check"].values,
                                x["scores"].values,
                            )
                        )
                        .mean()
                    )
                    areas.append(
                        {
                            "attr_level": attr,
                            "dataset": dataset,
                            "explanation_method": e,
                            "perturbations": p,
                            "baselines": b,
                            "area_random_exp": area,
                            "area_above_curve_under_sanity": area_above_curve_under_sanity,
                        }
                    )

area_df = pd.DataFrame(areas)
print("\nLOCAL")
print(
    area_df.query("attr_level=='local'")
    .groupby(["perturbations"])[["area_above_curve_under_sanity"]]
    .mean()
    .sort_values("area_above_curve_under_sanity")
)

print(
    area_df.query("attr_level=='local'")
    .groupby(["baselines"])[["area_random_exp"]]
    .mean()
    .sort_values("area_random_exp")
)


print("\nGLOBAL")
print(
    area_df.query("attr_level=='global'")
    .groupby(["perturbations"])[["area_above_curve_under_sanity"]]
    .mean()
    .sort_values("area_above_curve_under_sanity")
)
print(
    area_df.query("attr_level=='global'")
    .groupby(["baselines"])[["area_random_exp"]]
    .mean()
    .sort_values("area_random_exp")
)
