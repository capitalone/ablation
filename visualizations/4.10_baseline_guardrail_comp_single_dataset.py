# Imports
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D

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
    dataset = "spambase"

    attributions = "global"
    model = "nn"
    results_file = f"results-{attributions}.pkl"
    sanity_check_file = "model_sanity_checks.pkl"

    output_path = "Plots"
    os.makedirs(output_path, exist_ok=True)
    output_file = f"10_baseline_guardrail_comp_single_dataset_{dataset}.png"

    n_rows = 2
    n_columns = 2

    width = n_columns * 4
    height = n_rows * 3
    fig, ax = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey=True, figsize=(width, height)
    )
    colors = np.array(sns.color_palette("colorblind"))
    for i, a_base in enumerate(baselines):
        plt.subplot(221 + i)
        if dataset == "synthetic_cat":
            folder_path = f"../kdd_configs/kdd_experiments/{dataset}-nn-{attributions}"
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

        if i == len(baselines) - 1:
            g = sns.lineplot(
                data=tmp,
                x="pct_steps",
                y="scores",
                palette="colorblind",
                color=colors[i],
                label=str(a_base).replace("_", " "),
                legend=False,
            )
        else:
            g = sns.lineplot(
                data=tmp, x="pct_steps", y="scores", color=colors[i], legend=False,
            )

        g.set(xlabel=None)
        g.set(ylabel=None)

        color = np.array(sns.color_palette("colorblind"))

        x = sub_df[(sub_df["baseline"] == a_base)]["random_sanity_check_perc"].mean()

        g.axvline(
            x=sub_df[(sub_df["baseline"] == a_base)]["random_sanity_check_perc"].mean(),
            ls="--",
            c=colors[i],
        )

        base_label = a_base.replace("_", " ")
        g.set_title(f"{base_label}", y=1.02, ha="center")

    fig.supxlabel("Fraction of Features Ablated", y=0.01)
    fig.supylabel("AUROC")

    lines = [Line2D([0], [0], color=c) for c in colors[: len(baselines)]]
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, output_file), bbox_inches="tight")
    plt.close("all")
