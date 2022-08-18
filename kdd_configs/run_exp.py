import os
import pickle
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from ablation.experiment import Config, Experiment


def run(run_path, name):
    config = Config.from_yaml_file("config.yml")
    e = Experiment(config)

    df = e.run_exp()

    for scoring_method in config.ablation_args["scoring_methods"]:
        sub_df = df.query(f"score_name=='{scoring_method}'")
        fig_object = plt.figure()
        g = sns.FacetGrid(
            sub_df,
            col="perturbation",
            row="explanation_method",
            hue="baseline",
            palette="colorblind",
            margin_titles=True,
        )
        g.map(sns.lineplot, "pct_steps", "scores", palette="colorblind")
        g.add_legend()
        for row, exp in enumerate(sub_df.explanation_method.unique()):
            for col, perturb in enumerate(sub_df.perturbation.unique()):
                g.axes[row][col].axvline(
                    x=sub_df[
                        (sub_df.explanation_method == exp)
                        & (sub_df.perturbation == perturb)
                        & (sub_df.baseline != "random explanation")
                    ]["random_sanity_check_perc"].mean(),
                    ls="--",
                    c="red",
                    label="random feature",
                )

        g.map(
            plt.axhline,
            y=e.model_sanity_checks["label_shuffled"][scoring_method],
            ls="--",
            c="red",
            label="label_shuffed",
        )
        g = g.set(xlabel="Percent of features ablated", ylabel=scoring_method)
        g.fig.suptitle(name, y=1.02)
        g.savefig(
            os.path.join(
                run_path,
                f"{config.dataset_name}_ablation_{scoring_method}.png",
            ),
            bbox_inches="tight",
        )
        with open(
            os.path.join(
                run_path,
                f"{config.dataset_name}_ablation_{scoring_method}.pickle",
            ),
            "wb",
        ) as f:
            pickle.dump(fig_object, f)


if __name__ == "__main__":
    run_path_base = "kdd_experiments"
    if not (os.path.exists(run_path_base)):
        os.mkdir(run_path_base)

    # Parameters
    datasets = ["german", "spambase", "har", "adult"]
    sample_perc = [1, 1, 0.5, 0.5]
    models = ["nn"]
    is_local = [True, False]

    # Datasets
    for dd, data in enumerate(datasets):
        # Local or Global
        for explanation_type in is_local:
            # Models
            for mm, model in enumerate(models):

                if explanation_type:
                    name = f"{datasets[dd]}-{model}-local"
                else:
                    name = f"{datasets[dd]}-{model}-global"

                run_path = os.path.join(run_path_base, name)
                if not (os.path.exists(run_path)):
                    os.mkdir(run_path)

                print(run_path + "\n" + "***********************************")

                # Create Config
                with open("config.yml") as f:
                    con = yaml.safe_load(f)
                con["dataset_name"] = data
                con["dataset_sample_perc"] = sample_perc[dd]
                con["path"] = run_path
                con["model_type"] = model
                con["ablation_args"]["local"] = explanation_type
                with open("config.yml", "w") as f:
                    yaml.dump(con, f)

                run(run_path, name)
