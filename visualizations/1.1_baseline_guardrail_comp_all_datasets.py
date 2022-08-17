# Imports
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os


if __name__ == "__main__":

    explanation_methods = [
        "deep_shap",
    ]
    perturbations = ["constant_median"]
    baselines = [
        "nearest_neighbors",
        "constant_median",
        "opposite_class",
        "training",
    ]
    scores = ["auroc"]
    datasets = ["adult", "german", "har", "spambase"]

    attributions = "global"
    model = "nn"
    results_file = f"results-{attributions}.pkl"
    sanity_check_file = "model_sanity_checks.pkl"

    output_path = "Plots"
    os.makedirs(output_path, exist_ok=True)
    output_file = "1_teaser_image.png"

    fig, ax = plt.subplots(ncols=4, sharex=True, figsize=(12, 2.5))
    for i, a_dataset in enumerate(datasets):
        dataset = a_dataset
        if dataset == "synthetic_cat":
            folder_path = (
                f"../kdd_configs/kdd_experiments/{dataset}-linear-{attributions}"
            )
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
            tmp = sub_df[(sub_df["baseline"] == a_base)]
            x = tmp["pct_steps"]
            y = tmp["scores"]

            if i == len(datasets) - 1:
                g = sns.lineplot(
                    data=tmp,
                    x="pct_steps",
                    y="scores",
                    palette="colorblind",
                    ax=ax[i],
                    label=a_base.replace("_", " "),
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

        color = np.array(sns.color_palette("colorblind"))

        for i, a_base in enumerate(baselines):

            g.axvline(
                x=sub_df[(sub_df["baseline"] == a_base)][
                    "random_sanity_check_perc"
                ].mean(),
                ls="--",
                c=color[i],
            )

        g.axhline(y=sanity_df["label_shuffled"][scores[0]], ls="--", c="red")
        g.set_title(f"{dataset}", y=1.02, ha="center")

    fig.supxlabel("Fraction of Features Ablated", y=0.01)
    fig.supylabel("AUROC")

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.545, -0.03),
        fancybox=True,
        shadow=False,
        ncol=5,
    )

    fig.text(0.21, -0.13, "Baselines:")

    fig.tight_layout()
    plt.savefig(os.path.join(output_path, output_file), bbox_inches="tight")
    plt.close("all")
