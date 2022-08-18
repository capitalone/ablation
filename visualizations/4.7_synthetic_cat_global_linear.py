# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

if __name__ == "__main__":

    explanation_methods = ["deep_shap"]
    perturbations = ["constant_median", "marginal", "max_distance"]
    baselines = [
        "constant_median",
        "nearest_neighbors",
        "opposite_class",
        "training",
        "random explanation",
    ]
    scores = ["auroc"]
    dataset = "synthetic_cat"

    output_path = "Plots"
    if not (os.path.exists(output_path)):
        os.mkdir(output_path)
    attributions = "global"
    model = "linear"
    results_file = f"results-{attributions}.pkl"
    sanity_check_file = "model_sanity_checks.pkl"

    n_rows = 2
    n_columns = 2
    width = n_columns * 4
    height = n_rows * 3
    fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(width, height))
    plt.subplots_adjust(wspace=0.15, hspace=0.45)
    for i, pert in enumerate(perturbations):
        plt.subplot(221 + i)
        folder_path = f"../kdd_configs/kdd_experiments/{dataset}-{model}-{attributions}"
        results_df = pd.read_pickle(os.path.join(folder_path, results_file))
        sanity_df = pd.read_pickle(os.path.join(folder_path, sanity_check_file))

        sub_df = results_df.query(
            f"score_name in {scores} & explanation_method in {explanation_methods} & \
                        perturbation in {[pert]} & baseline in {baselines}"
        )

        for a_base in baselines:
            tmp = sub_df[sub_df["baseline"] == a_base]
            x = tmp["pct_steps"]
            y = tmp["scores"]
            if i == 2:
                g = sns.lineplot(
                    data=tmp,
                    x="pct_steps",
                    y="scores",
                    palette="colorblind",
                    label=str(a_base.replace("_", " ")),
                    legend=False,
                )
            elif i == 1:
                g = sns.lineplot(
                    data=tmp,
                    x="pct_steps",
                    y="scores",
                    palette="colorblind",
                    legend=False,
                )
            elif i == 0:
                g = sns.lineplot(
                    data=tmp,
                    x="pct_steps",
                    y="scores",
                    palette="colorblind",
                    legend=False,
                )
            g.set(xlabel="Fraction of Features Ablated")
            g.set(ylabel="AUROC")
            g.set_ylim([0, 1])
            g.set_yticks([0.2, 0.4, 0.6, 0.8])
            g.set_xticks([0.00, 0.25, 0.5, 0.75, 1.00])

        g.axvline(
            x=sub_df[(sub_df["baseline"] != "random explanation")][
                "random_sanity_check_perc"
            ].mean(),
            ls="--",
            c="red",
        )
        g.axhline(y=sanity_df["label_shuffled"][scores[0]], ls="--", c="red")
        g.set_title(pert.replace("_", " "), y=1.02, ha="center")

        plt.xticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax[-1, -1].axis("off")

    plt.legend(loc="center", bbox_to_anchor=(1.65, 0.5))
    plt.savefig(
        os.path.join(output_path, f"7_{dataset}_{attributions}_{model}.png"),
        bbox_inches="tight",
    )
