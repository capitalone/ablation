# plot three sanity checks as separate single axes plots
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

explanation_methods = ["deep_shap"]
perturbations = ["marginal"]
baselines = ["training", "random explanation"]
scores = ["auroc"]
dataset = "synthetic_cat"


output_path = "Plots"
os.makedirs(output_path, exist_ok=True)

attributions = "global"
model = "nn"
results_file = f"results-{attributions}.pkl"
sanity_check_file = "model_sanity_checks.pkl"


#
#  ------------ plot random explanation -------------
#
output_file = "4c_sanity_check_random_exp.png"
fig, ax = plt.subplots(ncols=1, sharex=False, figsize=(4, 3))

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
    g = sns.lineplot(
        data=tmp,
        x="pct_steps",
        y="scores",
        palette="colorblind",
        ax=ax,
        label=a_base,
        legend=False,
    )
    g.set(xlabel=None)
    g.set(ylabel=None)

ax.set_xlabel("Fraction of Features Ablated")
ax.set_ylabel("AUROC")

fig.tight_layout()
plt.savefig(os.path.join(output_path, output_file), bbox_inches="tight")
plt.close("all")
