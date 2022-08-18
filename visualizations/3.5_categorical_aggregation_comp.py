# Imports
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

if __name__ == "__main__":
    explanation_methods = [
        "deep_shap",
    ]
    perturbations = ["constant_median"]
    baselines = ["training", "random_explanation", "opposite_class"]
    scores = ["auroc"]
    datasets = [
        "german_non_agg",
        "german",
    ]

    output_path = "Plots"
    if not (os.path.exists(output_path)):
        os.mkdir(output_path)
    attributions = "global"
    model = "nn"
    results_file = f"results-{attributions}.pkl"
    sanity_check_file = "model_sanity_checks.pkl"

    n_rows = 1
    n_columns = len(datasets)

    width = n_columns * 4
    height = n_rows * 3

    fig, ax = plt.subplots(ncols=len(datasets), sharey=True, figsize=(width, height))
    for i, a_dataset in enumerate(datasets):
        dataset = a_dataset
        if a_dataset == "german_non_agg":
            dataset = "german"
            folder_path = f"../kdd_configs/kdd_experiments/{dataset}-{model}-{attributions}-non-agg"
        else:
            folder_path = (
                f"../kdd_configs/kdd_experiments/{dataset}-{model}-{attributions}"
            )
        results_df = pd.read_pickle(os.path.join(folder_path, results_file))
        sanity_df = pd.read_pickle(os.path.join(folder_path, sanity_check_file))

        sub_df = results_df.query(
            f"score_name in {scores} & explanation_method in {explanation_methods} & \
                perturbation in {perturbations} & baseline in {baselines}"
        )

        for a_base in baselines:
            tmp = sub_df[sub_df["baseline"] == a_base]
            x = tmp["pct_steps"]
            y = tmp["scores"]
            if i == len(datasets) - 1:
                g = sns.lineplot(
                    data=tmp,
                    x="pct_steps",
                    y="scores",
                    palette="colorblind",
                    ax=ax[i],
                    label=str(a_base.replace("_", " ")),
                    legend=False,
                )
            else:
                g = sns.lineplot(
                    data=tmp,
                    x="pct_steps",
                    y="scores",
                    palette="colorblind",
                    ax=ax[i],
                    legend=False,
                )
            ax[i].label_outer()
            g.set(xlabel=None)
            g.set(ylabel="AUROC")

        g.axvline(
            x=sub_df[(sub_df["baseline"] != "random explanation")][
                "random_sanity_check_perc"
            ].mean(),
            ls="--",
            c="red",
        )
        g.axhline(y=sanity_df["label_shuffled"][scores[0]], ls="--", c="red")

        if a_dataset == "german_non_agg":
            a_dataset = "no aggregation"
        if a_dataset == "german":
            a_dataset = "aggregated"

        g.set_title(f"{a_dataset}", y=1.02, ha="center")

    fig.supxlabel("Fraction of Features Ablated", y=0.01)
    plt.legend(loc="best")
    fig.tight_layout()
    plt.savefig(
        os.path.join(output_path, "5_aggregation_comp.png"), bbox_inches="tight"
    )
    plt.close("all")
