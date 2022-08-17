# Imports
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np
import pickle

import os

if __name__ == "__main__":
    explanation_methods = ["deep_shap"]
    perturbations = ["constant_median", "marginal"]
    baseline = "training"
    scores = ["auroc"]
    dataset = "adult"

    output_path = "Plots"
    if not (os.path.exists(output_path)):
        os.mkdir(output_path)
    attributions = ["global", "local"]
    model = "nn"
    results_file = "results"
    sanity_check_file = "model_sanity_checks.pkl"

    n_rows = 1
    n_columns = 2
    width = n_columns * 4
    height = n_rows * 3
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(width, height))
    plt.subplots_adjust(wspace=0.15, hspace=0.45)
    for i, pert in enumerate(perturbations):
        for attr in attributions:
            folder_path = f"../kdd_configs/kdd_experiments/{dataset}-{model}-{attr}"
            results_df = pd.read_pickle(os.path.join(folder_path, f"{results_file}-{attr}.pkl"))
            sanity_df = pd.read_pickle(os.path.join(folder_path, sanity_check_file))

            sub_df = results_df.query(
                f"score_name in {scores} & explanation_method in {explanation_methods} & \
                            perturbation in {[pert]} & baseline in {[baseline]}"
            )
            tmp = sub_df[sub_df["baseline"] == baseline]
            x = tmp["pct_steps"]
            y = tmp["scores"]
            if i == len(perturbations) - 1:
                g = sns.lineplot(
                    data=tmp,
                    x="pct_steps",
                    y="scores",
                    palette="colorblind",
                    ax=ax[i],
                    label=attr,
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
            g.set(xlabel=None)
            g.set(ylabel=None)

        g.axvline(
            x=sub_df[(sub_df["baseline"] != "random explanation")][
                "random_sanity_check_perc"
            ].mean(),
            ls="--",
            c="red",
        )
        g.axhline(y=sanity_df["label_shuffled"][scores[0]], ls="--", c="red")
        title_name = pert.replace("_", " ")
        g.set_title(f"{title_name}", y=1.02, ha="center")

        plt.xticks([0.00, 0.25, 0.50, 0.75, 1.00])

    fig.supxlabel("Fraction of Features Ablated", y=0.01)
    fig.supylabel("AUROC")
    ax[1].legend(loc="best")
    fig.tight_layout()
    plt.savefig(
        os.path.join(output_path, f"12_local_global_{dataset}_{baseline}_{model}.png"),
        bbox_inches="tight",
    )
    plt.close("all")
