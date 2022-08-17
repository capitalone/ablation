"""
data_loader.py
About: Load datasets for Ablation
"""
from dataclasses import dataclass
from os import path
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = path.join(path.dirname(path.abspath(__file__)), "data")


@dataclass
class NumpyDataset:
    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array
    n_classes: int

    # Feature names (pre and post encoding, respectively)
    feature_names: List[str]
    original_feature_names: List[str]

    def __post_init__(self):
        """
        Populate necessary fields after dataclass initialization.  In this
        case, generate the aggregation mapping.
        """
        self._agg_map = self.calculate_aggregation_map()

    def stratified_subsample(self, percentage, random_state=42):
        _, self.X_train, _, self.y_train = train_test_split(
            self.X_train,
            self.y_train,
            test_size=int(len(self.y_train) * percentage),
            random_state=random_state,
            stratify=self.y_train,
        )
        _, self.X_test, _, self.y_test = train_test_split(
            self.X_test,
            self.y_test,
            test_size=int(len(self.y_test) * percentage),
            random_state=random_state,
            stratify=self.y_test,
        )

    def add_random_features(self, n_rand_features=4, random_state=42):

        np.random.seed(random_state)
        random_train = np.random.normal(0, 1, (len(self.X_train), n_rand_features))
        random_test = np.random.normal(0, 1, (len(self.X_test), n_rand_features))
        self.X_train = np.concatenate([self.X_train, random_train], -1)
        self.X_test = np.concatenate([self.X_test, random_test], -1)

        # Calculate the random feature names
        random_feature_names = [f"#RANDOM{idx}#" for idx in range(n_rand_features)]

        # Add random feature names to list of post-encoded features
        self.feature_names += random_feature_names

        # Add random feature names to list of original features
        self.original_feature_names += random_feature_names

        # Re-calculate map since features have been added
        self._agg_map = self.calculate_aggregation_map()

    def calculate_aggregation_map(self) -> Union[List[List[int]], None]:
        """
        Generate a list of indices to map sparse to dense.  If the feature sets match
        no mapping can be created (except for 1-to-1).  Return None and downstream
        processes should interpret that as there being no difference in pre and post
        encoded features.
        """
        if len(self.feature_names) == len(self.original_feature_names):
            return None
        else:
            return [
                [
                    j
                    for (j, val) in enumerate(self.feature_names)
                    if ((x == val) or (x == "_".join(val.split("_")[:-1])))
                ]
                for (i, x) in enumerate(self.original_feature_names)
            ]

    @property
    def agg_map(self):
        return self._agg_map

    @property
    def random_feat_idx(self):
        return np.array(
            [idx for (idx, val) in enumerate(self.feature_names) if "#RANDOM" in val]
        )

    @property
    def dense_random_feat_idx(self):
        return np.array(
            [
                idx
                for (idx, val) in enumerate(self.original_feature_names)
                if "#RANDOM" in val
            ]
        )


def load_data(
    dataset: str,
    dataset_sample_percentage: float = 1,
    n_rand_features: int = 0,
) -> NumpyDataset:
    """Load preprocessed data from the list

    Args:
        dataset (str): Dataset name to be loaded
        dataset_sample_percentage (float): stratified subsample of dataset
        n_rand_features (int): number of random features to append for sanity checking explanation

    Returns:
        NumpyDataset: dataset
    """

    options = {
        "german": prepare_german_data,
        "adult": prepare_adult_data,
        "spambase": prepare_spambase_data,
        "har": prepare_har_data,
        "synthetic": prepare_synthetic_data,
        "synthetic_multiclass": prepare_synthetic_multiclass_data,
        "synthetic_cat": prepare_synthetic_categorical_data,
    }
    dataset = options[dataset]()
    if dataset_sample_percentage < 1:
        dataset.stratified_subsample(dataset_sample_percentage)
    if n_rand_features > 0:
        dataset.add_random_features(n_rand_features)
    return dataset


def prepare_german_data() -> NumpyDataset:
    """Prepare German dataset

    Returns:
        NumpyDataset: dataset
    """

    data = pd.read_csv(path.join(DATA_PATH, "MyGermanData.csv"))
    X, y = data.drop("credit.rating", axis=1), data["credit.rating"]

    cat_ix = X.select_dtypes(include=["object"]).columns
    num_ix = X.select_dtypes(include=["int64"]).columns

    X_train, y_train, X_test, y_test = split_dataset(
        X, y.values.flatten(), test_perc=0.2
    )

    encoder = OneHotEncoder()
    scaler = StandardScaler()

    ct = ColumnTransformer(
        [("categoricals", encoder, cat_ix), ("numericals", scaler, num_ix)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    return NumpyDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=2,
        feature_names=list(ct.get_feature_names_out()),
        original_feature_names=cat_ix.tolist() + num_ix.tolist(),
    )


def prepare_adult_data() -> NumpyDataset:
    """Prepare Adult dataset

    Returns:
        NumpyDataset: dataset
    """

    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income-over-50k",
    ]

    training_data = pd.read_csv(
        path.join(DATA_PATH, "adult.data"), names=column_names, index_col=False
    )
    test_data = pd.read_csv(
        path.join(DATA_PATH, "adult.test"), names=column_names, index_col=False
    )

    training_data["income-over-50k"] = training_data["income-over-50k"].map(
        {" >50K": 1, " <=50K": 0}
    )
    test_data["income-over-50k"] = test_data["income-over-50k"].map(
        {" >50K": 1, " <=50K": 0}
    )

    X_train = training_data.drop("income-over-50k", axis=1)
    y_train = training_data["income-over-50k"]
    X_test = test_data.drop("income-over-50k", axis=1)
    y_test = test_data["income-over-50k"]

    cat_ix = X_train.select_dtypes(include=["object"]).columns
    num_ix = X_train.select_dtypes(include=["int64"]).columns

    encoder = OneHotEncoder()
    scaler = StandardScaler()

    ct = ColumnTransformer(
        [("categoricals", encoder, cat_ix), ("numericals", scaler, num_ix)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    return NumpyDataset(
        X_train=X_train.toarray(),
        y_train=y_train.values.flatten(),
        X_test=X_test.toarray(),
        y_test=y_test.values.flatten(),
        n_classes=2,
        feature_names=list(ct.get_feature_names_out()),
        original_feature_names=cat_ix.tolist() + num_ix.tolist(),
    )


def prepare_spambase_data() -> NumpyDataset:
    """Prepare Spambase dataset

    Returns:
        NumpyDataset: dataset
    """
    X = pd.read_csv(path.join(DATA_PATH, "spambase-X.csv"))
    y = pd.read_csv(path.join(DATA_PATH, "spambase-y.csv"))

    X_train, y_train, X_test, y_test = split_dataset(
        X.values, y.values.flatten(), test_perc=0.2
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return NumpyDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=len(np.unique(y)),
        feature_names=list(X.columns),
        original_feature_names=list(X.columns),
    )


def prepare_har_data() -> NumpyDataset:
    """Prepare HAR dataset

    Returns:
        NumpyDataset: dataset
    """
    # Load Datasets
    x_train = pd.read_pickle(path.join(DATA_PATH, "har_x_train.pkl"))
    y_train = pd.read_pickle(path.join(DATA_PATH, "har_y_train.pkl"))
    x_test = pd.read_pickle(path.join(DATA_PATH, "har_x_test.pkl"))
    y_test = pd.read_pickle(path.join(DATA_PATH, "har_y_test.pkl"))

    # Load Columns
    feature_names = pd.read_table(
        path.join(DATA_PATH, "har_features.txt"),
        sep="\s+",
        header=None,
        squeeze=True,
    )
    feature_names = feature_names[1].values

    # Standard Scaler
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # # Convert y to numpy array
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Change number from 0 to n_classes -1
    y_train -= 1
    y_test -= 1

    return NumpyDataset(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        n_classes=len(np.unique(y_train)),
        feature_names=list(feature_names),
        original_feature_names=list(feature_names),
    )


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_perc: float = 0.0,
    test_perc: float = 0.2,
):

    assert test_perc > 0, "Must have a test set"

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=test_perc, random_state=42)

    if val_perc > 0:
        (X_train, X_val, y_train, y_val,) = train_test_split(
            X_train,
            y_train,
            test_size=val_perc / (1 - test_perc),
            random_state=42,
        )
        return X_train, y_train, X_val, y_val, X_test, y_test
    return X_train, y_train, X_test, y_test


def prepare_synthetic_data() -> NumpyDataset:
    """Prepare synthetic dataset

    Returns:
        NumpyDataset: dataset
    """
    np.random.seed(42)
    d = 20
    n = 1000
    X = np.random.normal(size=(n, d))
    coef = np.random.normal(size=(d, 1))
    prob = 1 / (1 + np.exp(-X @ coef))
    y = (prob > np.random.rand(n, 1)).astype(int).squeeze()
    X_train, y_train, X_test, y_test = split_dataset(X, y, test_perc=0.2)
    return NumpyDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=len(np.unique(y)),
        feature_names=[f"feat_{i}" for i in range(X_train.shape[1])],
        original_feature_names=[f"feat_{i}" for i in range(X_train.shape[1])],
    )


def prepare_synthetic_multiclass_data() -> NumpyDataset:
    """Prepare synthetic dataset

    Returns:
        NumpyDataset: dataset
    """
    np.random.seed(42)
    d = 20
    n = 1000
    n_classes = 3
    X = np.random.normal(size=(n, d))
    coef = np.random.normal(size=(d, n_classes))
    prob = np.exp(X @ coef) / np.exp(X @ coef).sum(-1, keepdims=True)
    y = np.array([np.random.choice(list(range(n_classes)), p=p) for p in prob])
    X_train, y_train, X_test, y_test = split_dataset(X, y, test_perc=0.2)
    return NumpyDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=len(np.unique(y)),
        feature_names=[f"feat_{i}" for i in range(X_train.shape[1])],
        original_feature_names=[f"feat_{i}" for i in range(X_train.shape[1])],
    )


def prepare_synthetic_categorical_data() -> NumpyDataset:
    """Prepare synthetic dataset

    Returns:
        NumpyDataset: dataset
    """
    np.random.seed(42)

    d = 20
    n = 1000
    n_cat = 5
    n_cont = d - n_cat
    levels = 6

    X_cont = np.random.normal(size=(n, n_cont))
    X_cat = (np.random.rand(n, n_cat) * levels).astype(int)
    X = np.hstack([X_cont, X_cat])
    coef_cont = np.random.normal(size=(n_cont, 1))
    coef_cat = np.random.normal(size=(1, n_cat, levels))

    X_cat_mult_coef = (np.eye(levels)[X_cat] * coef_cat).sum(axis=-1)
    logit_cat = X_cat_mult_coef.sum(axis=-1, keepdims=True)

    logit_cont = X_cont @ coef_cont
    logits = logit_cont + logit_cat
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > np.random.rand(n, 1)).astype(int).squeeze()

    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(d)])
    cat_feat = [f"feat_{i}" for i in range(n_cont, d)]
    cont_feat = [f"feat_{i}" for i in range(n_cont)]
    X_df[cat_feat] = X_df[cat_feat].astype(int)
    encoder = OneHotEncoder()
    scaler = StandardScaler()
    X_train, y_train, X_test, y_test = split_dataset(X_df, y, test_perc=0.2)
    ct = ColumnTransformer(
        [
            ("categoricals", encoder, cat_feat),
            ("numericals", scaler, cont_feat),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    return NumpyDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=len(np.unique(y)),
        feature_names=list(ct.get_feature_names_out()),
        original_feature_names=list(X_df.columns),
    )
