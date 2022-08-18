# plot three sanity checks as separate single axes plots
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

explanation_methods = ["deep_shap"]
perturbations = ["marginal"]
baselines = ["training"]
scores = ["auroc"]
dataset = "synthetic_cat"


output_path = "Plots"
os.makedirs(output_path, exist_ok=True)

attributions = "global"
model = "nn"
results_file = f"results-{attributions}.pkl"
sanity_check_file = "model_sanity_checks.pkl"


folder_path = f"../kdd_configs/kdd_experiments/{dataset}-{model}-{attributions}"
results_df = pd.read_pickle(os.path.join(folder_path, results_file))
sanity_df = pd.read_pickle(os.path.join(folder_path, sanity_check_file))

sub_df = results_df.query(
    f"score_name in {scores} & explanation_method in {explanation_methods} & \
                perturbation in {perturbations} & baseline in {baselines}"
)

vertical_guardrail = sub_df[(sub_df["baseline"] != "random explanation")][
    "random_sanity_check_perc"
].mean()
horizontal_guardrail = sanity_df["label_shuffled"][scores[0]]


#
#  ------------ plot worst case model performance -------------
#
output_file = "4a_sanity_check_worst_case_model.png"
fig, ax = plt.subplots(ncols=1, sharex=False, figsize=(4, 3))

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
g.axhline(y=horizontal_guardrail, ls="--", c="red")

fig.tight_layout()
plt.savefig(os.path.join(output_path, output_file), bbox_inches="tight")
plt.close("all")

#
#  ------------ plot random feature  vertical line -------------
#
output_file = "4b_sanity_check_random_feature.png"
fig, ax = plt.subplots(ncols=1, sharex=False, figsize=(4, 3))

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
g.axvline(
    x=vertical_guardrail, ls="--", c="red",
)

fig.tight_layout()
plt.savefig(os.path.join(output_path, output_file), bbox_inches="tight")
plt.close("all")


#
#  ------------ plot both with quadrants -------------
#
output_file = "4d_sanity_check_quadrants.png"
fig, ax = plt.subplots(ncols=1, sharex=False, figsize=(4, 3))

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

ylim = (horizontal_guardrail - (1 - horizontal_guardrail) / 2, 1)
ax.set_ylim(ylim)
ax.set_xlabel("Fraction of Features Ablated")
ax.set_ylabel("AUROC")

g.axvline(x=vertical_guardrail, ls="--", c="red")
g.axhline(y=horizontal_guardrail, ls="--", c="red")
ax.annotate(
    "I",
    (
        vertical_guardrail + (1 - vertical_guardrail) / 2,
        horizontal_guardrail + (1 - horizontal_guardrail) / 2,
    ),
    fontsize=12,
)
ax.annotate(
    "II",
    (vertical_guardrail / 2, horizontal_guardrail + (1 - horizontal_guardrail) / 2,),
    fontsize=12,
)
ax.annotate(
    "III",
    (vertical_guardrail / 2, horizontal_guardrail - horizontal_guardrail / 3),
    fontsize=12,
)
ax.annotate(
    "IV",
    (
        vertical_guardrail + (1 - vertical_guardrail) / 2,
        horizontal_guardrail - horizontal_guardrail / 3,
    ),
    fontsize=12,
)


fig.tight_layout()
plt.savefig(os.path.join(output_path, output_file), bbox_inches="tight")
plt.close("all")
