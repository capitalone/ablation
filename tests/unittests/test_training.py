import shutil

from ablation.dataset import load_data
from ablation.pytorch_model import train
from ablation.utils.evaluate import eval_model_performance


def test_dataset_training():
    DATASETS = [
        "german",
        "adult",
        "spambase",
        "har",
        "synthetic",
        "synthetic_multiclass",
        "synthetic_cat",
    ]

    for name in DATASETS:
        data = load_data(name, dataset_sample_percentage=0.2)
        train(data, max_epochs=1, path="./test_logs")

    shutil.rmtree("./test_logs", ignore_errors=True)


def test_label_shuffled_training():
    data = load_data("synthetic", dataset_sample_percentage=0.3)
    model = train(data, max_epochs=1, path="./test_logs")
    label_shuffed_model = train(
        data, path="./test_logs_shuffled", max_epochs=1, shuffle_labels=True
    )

    shutil.rmtree("./test_logs", ignore_errors=True)
    shutil.rmtree("./test_logs_shuffled", ignore_errors=True)

    model_loss = eval_model_performance(
        model, data.X_test, data.y_test, scoring_method="log_loss"
    )
    label_shuffed_model_loss = eval_model_performance(
        label_shuffed_model,
        data.X_test,
        data.y_test,
        scoring_method="log_loss",
    )
    assert model_loss < label_shuffed_model_loss
