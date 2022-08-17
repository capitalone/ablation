# Imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

if __name__ == "__main__":
    explanation_methods = [
        "random",
        "deep_shap",
        "kernel_shap",
        "integrated_gradients",
    ]
    perturbations = ["constant_median"]
    baselines = ["training"]
    scores = ["auroc"]
    datasets = [
        "synthetic_cat",
        "adult",
        "german",
        "har",
        "spambase",
    ]

    output_path = "Plots"
    if not (os.path.exists(output_path)):
        os.mkdir(output_path)
    attributions = "global"
    model = "nn"
    results_file = f"results-{attributions}.pkl"
    sanity_check_file = "model_sanity_checks.pkl"

    n_rows = 3
    n_columns = 2

    width = n_columns * 4
    height = n_rows * 3

    fig, ax = plt.subplots(ncols=2, nrows=3, sharey=False, figsize=(width, height))

    plt.subplots_adjust(wspace=0.15, hspace=0.45)
    for i, a_dataset in enumerate(datasets):
        plt.subplot(321 + i)
        dataset = a_dataset
        folder_path = f"../kdd_configs/kdd_experiments/{dataset}-{model}-{attributions}"
        results_df = pd.read_pickle(os.path.join(folder_path, results_file))
        sanity_df = pd.read_pickle(os.path.join(folder_path, sanity_check_file))

        sub_df = results_df.query(
            f"score_name in {scores} & explanation_method in {explanation_methods} & \
                        perturbation in {perturbations} & baseline in {baselines}"
        )

        for a_exp in explanation_methods:
            tmp = sub_df[sub_df["explanation_method"] == a_exp]
            x = tmp["pct_steps"]
            y = tmp["scores"]

            g = sns.lineplot(
                data=tmp,
                x="pct_steps",
                y="scores",
                palette="colorblind",
                label=str(a_exp.replace("_", " ")),
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
        if dataset == "synthetic_cat":
            g.set_title(f"synthetic", y=1.02, ha="center")
        else:
            g.set_title(f"{dataset}", y=1.02, ha="center")

        ax[-1, -1].axis("off")

    plt.legend(loc="center", bbox_to_anchor=(1.65, 0.5))
    plt.savefig(
        os.path.join(output_path, "13_explanation_comp.png"), bbox_inches="tight"
    )
    plt.close("all")
