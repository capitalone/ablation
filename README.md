
# Ablation

A library to assess the effectiveness of XAI methods through ablation.

For API documentation, visit the [documentation page](https://capitalone.github.io/ablation/).

### Background

Explainable artificial intelligence (XAI) methods lack ground truth.  In its place, method developers have relied on axioms to determine desirable properties for their explanations behavior.  For high stakes uses of machine learning that require explainability, it is not sufficient to rely on axioms, as the implementation, or its usage, can fail to live up to the ideal.  A procedure frequently used to assess their utility, and to some extent their fidelity, is an *ablation study*.  By perturbing the input variables in rank order of importance, the goal is to assess the sensitivity of the model's performance.

This implementation can be used to reproduce the experiments in [BASED-XAI: Breaking Ablation Studies Down for Explainable Artificial Intelligence](https://arxiv.org/abs/2207.05566).

### Installation

Pip

```sh
conda create -n ablation python=3.9 --yes
source activate ablation
pip install ablation
```

### Contributions

If you would like to contribute, please download the code from `master` to ensure any contributions are made towards vulnerability free code.

Install the development dependencies:
```
pip install -e ."[dev]"
```

Install `pre-commit` hooks:

```
pre-commit install
pre-commit run
```

### Get Started

Configure experiment yaml file:

```yaml
dataset_name: synthetic_cat             # dataset name
model_type: nn                          # model type (nn/linear)
path: synthetic_cat                     # experiment path
load: false                             # load experiment from path
rerun_ablation: false                # rerun ablation if loaded

n_trials: 4                             # number of trials
dataset_sample_perc: 1                  # fraction of dataset to subsample
dataset_n_random_features: 4            # random features to add

explanation_methods:                    # explanation methods
  - random
  - deep_shap
  - kernel_shap
perturbation_config:                    # perturbation methods and dict of params
  constant_median: {}

baseline_config:                        # baseline methods and dict of params
  constant_median: {}
  training:
    nsamples: 50


ablation_args:                          # ablation arguments
  scoring_methods: [log_loss, auroc]    # performance metric
  local: true                           # local vs global explanations

```

Run experiment
```Python
from ablation.experiment import Config, Experiment

config = Config.from_yaml_file("config.yml")
exp = Experiment(config)
results = exp.run_exp()
```
### Replicate experiments

Experiments for the paper used `pytorch_lightning==1.5.7`. Due to vulnerabilities, we needed to update to a more recent version.  In order to accurately reproduce the experiments in the paper, install release `v0.1.0` and revert to the previous version of pytorch lightning:
```
pip install ablation==0.1.0
pip install pytorch_lightning==1.5.7
```

To replicate KDD experiments for all real datasets run the following
```sh
cd kdd_configs
python run_exp.py
```

To replicate KDD experiments for synthetic datasets run the following
```sh
cd kdd_configs
python run_exp_synthetic.py
```

### Tests

To run tests:
```sh
python -m pytest tests/unittests
```

### Citing
If you use `ablation` in your work, please cite our paper:

```
@inproceedings{hameed:basedxai:2022,
  author    = {Hameed, Isha and Sharpe, Samuel and Barcklow, Daniel and Au-Yeung, Justin and Verma, Sahil and Huang, Jocelyn and Barr, Brian and Bruss, C. Bayan},
  title     = {{BASED-XAI: Breaking Ablation Studies Down for Explainable Artificial Intelligence}},
  year      = {2022},
  maintitle = {ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  booktitle = {Workshop on Machine Learning in Finance},
}
```

### Contributors

We welcome Your interest in Capital One’s Open Source Projects (the
“Project”). Any Contributor to the Project must accept and sign an
Agreement indicating agreement to the license terms below. Except for
the license granted in this Agreement to Capital One and to recipients
of software distributed by Capital One, You reserve all right, title,
and interest in and to Your Contributions; this Agreement does not
impact Your rights to use Your own Contributions for any other purpose.

[Sign the Individual Agreement](https://docs.google.com/forms/d/19LpBBjykHPox18vrZvBbZUcK6gQTj7qv1O5hCduAZFU/viewform)

[Sign the Corporate Agreement](https://docs.google.com/forms/d/e/1FAIpQLSeAbobIPLCVZD_ccgtMWBDAcN68oqbAJBQyDTSAQ1AkYuCp_g/viewform?usp=send_form)


### Code of Conduct 

This project adheres to the [Open Code of Conduct](https://developer.capitalone.com/resources/code-of-conduct)
By participating, you are
expected to honor this code.
