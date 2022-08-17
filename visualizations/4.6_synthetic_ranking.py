import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

synth_coef = pickle.load(open("synthetic_cat_coef.pkl", "rb"))

# Combine continuous and categorical coefficients
# Categoricals are evenly distributed between all levels,
# so give the mean across levels as our coef estimate
coef = np.concatenate(
    [
        synth_coef["coef_cont"].flatten(),
        synth_coef["coef_cat"].mean(-1).flatten(),
        np.array([0, 0, 0, 0]),  # random features
    ]
)
rank = np.arange(len(coef) + 1)
ordered_abs_value = np.sort(np.abs(coef))

importance_remaining = np.concatenate(
    [
        [np.sum(ordered_abs_value)],
        np.sum(ordered_abs_value) - np.cumsum(ordered_abs_value[::-1]),
    ]
)
plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
fig, ax = plt.subplots(ncols=1, sharex=False, figsize=(5, 3.1))
sns.scatterplot(
    rank, importance_remaining / np.sum(ordered_abs_value), ax=ax,
)
plt.xlabel("\nCoefficient Rank")
plt.ylabel("Fraction of Global\nImportance Remaining\n")
plt.xticks(np.arange(0, len(coef), 5))
plt.savefig(os.path.join("Plots", "6_synthetic_ranking.png"), bbox_inches="tight")
plt.close("all")
