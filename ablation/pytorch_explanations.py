from typing import Optional

import numpy as np
import torch
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import DeepLift, DeepLiftShap, IntegratedGradients, KernelShap, Lime
from captum.attr._core.lime import get_exp_kernel_similarity_function
from pytorch_lightning import seed_everything

from .baseline import (
    ConstantBaseline,
    ManyToOneBaseline,
    OneToOneBaseline,
    SampleBaseline,
)
from .utils.logging import timing
from .utils.model import _as_numpy, _predict_proba_fn, _torch_float, _torch_long

CAPTUM_METHODS = {
    "integrated_gradients": IntegratedGradients,
    "deep_lift": DeepLift,
    "deep_shap": DeepLiftShap,
    "kernel_shap": KernelShap,
    "lime": Lime,
}
SAMPLE_BASED_METHODS = ["deep_shap"]


@timing
def captum_explanation(
    method: str,
    model: torch.nn.Module,
    X: np.ndarray,
    baseline: np.ndarray,
    target: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    **kwargs,
):
    """Collection of captum based explanation methods following the
        On-Baselines paper.

        Baselines either are constant (one sample), have a one-to-one
        relationship with the observations (max_distance), or are a sample
        from a larger distribution that needs to be averaged over.

    Args:
        method (str): name of explanation method
        model (torch.nn.Module): torch model
        X (np.ndarray): Observations to derive explainations from
        baseline (np.ndarray): Baseline array.
        target (np.ndarray, optional): If multiclass, should be array of integers.
                If it is a binary problem it should be None. Defaults to None.
    """
    seed_everything(random_state)

    if method == "random":
        return np.random.rand(*X.shape)

    X = _torch_float(X)
    baseline_type = type(baseline)
    baseline = _torch_float(baseline)
    y_pred = model(X)

    if target is None and y_pred.size(1) > 1:
        # Use predicted class for multiclass
        target = y_pred.argmax(-1)
    elif target is not None:
        # Use target if provided
        target = _torch_long(target)

    if method not in CAPTUM_METHODS:
        raise ValueError(
            f"Method {method} not in available explanation methods. "
            f"""Choose one of "{'", "'.join(CAPTUM_METHODS.keys())}" """
        )

    explainer_args = {}

    if method == "lime":
        explainer_args.update(
            {
                "interpretable_model": SkLearnLinearRegression(),
                "similarity_func": get_exp_kernel_similarity_function(
                    distance_mode="euclidean",
                    kernel_width=np.sqrt(X.size(1)) * 0.75,
                ),
            }
        )
    if method == "deep_shap" and baseline_type in [
        ConstantBaseline,
        OneToOneBaseline,
        ManyToOneBaseline,
    ]:
        # Deep shap cannot be used for constant, one to one baselines, or many to one baselines
        method = "deep_lift"

    exp = CAPTUM_METHODS[method](model, **explainer_args)

    if baseline_type == ManyToOneBaseline:
        # Many to one baseline
        # baseline shape: (# observations in X, # baseline samples per observation, # features)
        attr = torch.zeros_like(X, device=X.device)
        for b in baseline.permute(1, 0, 2):
            attr += exp.attribute(inputs=X, baselines=b, target=target, **kwargs)
        return _as_numpy(attr / baseline.size(1))

    elif baseline_type == SampleBaseline and method not in SAMPLE_BASED_METHODS:
        # Non-sample based method and sample baseline
        # baseline shape: (# baseline samples, # features)
        attr = torch.zeros_like(X, device=X.device)
        for b in baseline:
            attr += exp.attribute(
                inputs=X, baselines=b.unsqueeze(0), target=target, **kwargs
            )
        return _as_numpy(attr / baseline.size(0))

    return _as_numpy(
        exp.attribute(inputs=X, baselines=baseline, target=target, **kwargs)
    )
