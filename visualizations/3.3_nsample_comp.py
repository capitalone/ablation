import pandas as pd
import numpy as np
import os
from os.path import join as jp, abspath, split as splitpath
import pickle
import yaml
import matplotlib.pyplot as plt
import scipy.stats as stats
import collections
import sys

if __name__ == "__main__":

    explanation_method = "deep_shap"
    baseline = "training"
    score = "auroc"
    dataset = "spambase"
    ntrials = 3

    nsamples = [10, 20, 50, 75, 100, 160, 200, 300, 400, 920, 1840]

    attributions = "global"
    model = "nn"
    explanations_file = "explanations.pkl"

    output_path = "Plots"
    os.makedirs(output_path, exist_ok=True)
    output_file = f"3_nsample_kendall_tau_comparison_{dataset}.png"

    global_rankings = {}

    for nsample_val in nsamples:
        folder_path = (
            f"../kdd_configs/nsample_runs/{dataset}_nn_n-samples_{nsample_val}"
        )
        explanations_df = pd.read_pickle(os.path.join(folder_path, explanations_file))

        trials = [
            explanations_df[trial][explanation_method][baseline]
            for trial in range(ntrials)
        ]

        trials_avg = np.mean(trials, axis=0)

        global_importances = np.abs(trials_avg).mean(0)

        global_rankings[nsample_val] = np.argsort(-global_importances).transpose()

    tau_values = {}
    p_values = {}

    global_rankings_max = global_rankings[nsamples[-1]].tolist()

    for nsample in global_rankings:
        global_rankings_current = global_rankings[nsample].tolist()

        x = list(global_rankings_current)
        y = list(global_rankings_max)

        mapping = {n: r for r, n in enumerate(x)}
        new_x = list(range(len(x)))
        new_y = [mapping[i] for i in y]

        tau_values[nsample], p_values[nsample] = stats.kendalltau(new_x, new_y)

    plt.figure(figsize=(8, 6))
    r = np.arange(len(nsamples))
    plt.bar(
        r,
        tau_values.values(),
        color="green",
        width=0.25,
        edgecolor="black",
        label=dataset,
    )
    plt.xlabel("Number of Samples in Training Baseline")
    plt.ylabel("Kendall's Tau Score")
    plt.xticks(r + 0.05, nsamples)
    plt.legend()

    plt.savefig(os.path.join(output_path, output_file), bbox_inches="tight")

    plt.close("all")
