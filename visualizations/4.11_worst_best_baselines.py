# Imports
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

if __name__ == "__main__":

    explanation_methods = ["random", "deep_shap"]
    perturbations = ["constant_median"]
    baselines = [
        "nearest_neighbors",
        "opposite_class",
        "random explanation",
    ]
    scores = ["auroc"]
    datasets = ["adult", "german", "har", "spambase"]

    attributions = "global"
    model = "nn"
    results_file = f"results-{attributions}.pkl"
    sanity_check_file = "model_sanity_checks.pkl"

    output_path = "Plots"
    os.makedirs(output_path, exist_ok=True)
    output_file = "11_best_worst_baseline.png"

    n_rows = 2
    n_columns = 2
    width = n_columns * 4
    height = n_rows * 3
    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=False, figsize=(width, height))
    plt.subplots_adjust(wspace=0.15, hspace=0.45)
    for i, a_dataset in enumerate(datasets):
        ax1 = plt.subplot(221 + i)
        dataset = a_dataset
        folder_path = f"../kdd_configs/kdd_experiments/{dataset}-{model}-{attributions}"
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
            if a_base == "constant_median":
                basename = "constant median"
            if a_base == "opposite_class":
                basename = "opposite class"
            if a_base == "nearest_neighbors":
                basename = "nearest neighbors"
            if a_base == "training":
                basename = "training"
            if a_base == "random explanation":
                basename = "random explanation"
            if i == 1:
                g = sns.lineplot(
                    data=tmp,
                    x="pct_steps",
                    y="scores",
                    palette="colorblind",
                    label=basename,
                    legend=False,
                )
            else:
                g = sns.lineplot(
                    data=tmp,
                    x="pct_steps",
                    y="scores",
                    palette="colorblind",
                    legend=False,
                )

            g.set(xlabel="Fraction of Features Ablated")
            if (i == 0) or (i == 2) or (i == 4):
                g.set(ylabel="AUROC")
            else:
                g.set(ylabel=None)
            g.set_xticks([0.00, 0.25, 0.5, 0.75, 1.00])

        g.axvline(
            x=sub_df[(sub_df["baseline"] != "random explanation")][
                "random_sanity_check_perc"
            ].mean(),
            ls="--",
            c="red",
        )
        g.axhline(y=sanity_df["label_shuffled"][scores[0]], ls="--", c="red")
        g.set_title(f"{dataset}", y=1.02, ha="center")
        if i == 1:
            ax1.legend(loc="best")

    fig.supxlabel("Fraction of Features Ablated", y=0.01)
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, output_file), bbox_inches="tight")
    plt.close("all")
