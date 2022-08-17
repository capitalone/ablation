"""
pytorch_model.py
About: Pytorch model and datamodule
"""

import os
from copy import copy
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import NumpyDataset, split_dataset
from .utils.logging import logger as exp_logger
from .utils.model import _as_numpy, _torch_float


class LinearModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.net = nn.Linear(input_size, n_classes)

    def forward(self, x):
        return self.net(x)


class NNModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_size * 2),
            nn.Dropout(p=0.25),
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(),
            nn.BatchNorm1d(input_size),
            nn.Dropout(p=0.25),
            nn.Linear(input_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class Classifier(LightningModule):
    def __init__(self, input_size, n_classes, model_type="nn"):
        super().__init__()
        self.save_hyperparameters()
        output_size = n_classes if self.hparams.n_classes > 2 else 1

        self.model = (
            NNModel(input_size, output_size)
            if model_type == "nn"
            else LinearModel(input_size, output_size)
        )
        self.criterion = (
            nn.CrossEntropyLoss() if self.hparams.n_classes > 2 else nn.BCELoss()
        )

    def forward(self, x):
        logits = self.model(x)
        if self.hparams.n_classes > 2:
            return torch.softmax(logits, -1)
        return torch.sigmoid(logits)

    def predict_numpy(self, x: np.ndarray):
        return _as_numpy(self.predict(_torch_float(x, device=self.device)))

    def _step(self, batch, batch_idx):
        x, y = batch
        prob = self.forward(x).squeeze(-1)
        loss = self.criterion(prob, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())


class TensorDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: NumpyDataset,
        batch_size: int = None,
        shuffle_labels: bool = False,
    ):
        """Data module for training models

        Args:
            dataset (NumpyDataset): numpy dataset
            batch_size (int): batch size. Defaults to None.
            shuffle_labels (bool): corrupt y labels by permuting them. Defaults to False.
        """
        super().__init__()
        self.n_classes = dataset.n_classes

        dataset = self._corrupt_dataset(dataset, shuffle_labels)
        X_train, y_train, X_val, y_val = split_dataset(
            dataset.X_train,
            dataset.y_train,
            test_perc=0.1,
        )
        self.batch_size = (
            int(np.sqrt(len(X_train))) if batch_size is None else batch_size
        )
        self.X_train, self.y_train = self.convert(X_train, y_train)
        self.X_val, self.y_val = self.convert(X_val, y_val)
        self.X_test, self.y_test = self.convert(dataset.X_test, dataset.y_test)

    def _corrupt_dataset(self, dataset, shuffle_labels):

        new_dataset = copy(dataset)

        if shuffle_labels:
            new_dataset.y_train = np.random.permutation(new_dataset.y_train)

        return new_dataset

    def convert(self, X, y):
        X = torch.tensor(X).float()
        y = torch.tensor(y)
        y = y.float() if self.n_classes == 2 else y.long()
        return X, y

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.batch_size,
        )


def train(
    data: NumpyDataset,
    path,
    max_epochs=100,
    model_type="nn",
    prefix="model",
    shuffle_labels=False,
    random_state=42,
    log=True,
):

    seed_everything(random_state)
    datamodule = TensorDataModule(dataset=data, shuffle_labels=shuffle_labels)
    model = Classifier(data.X_train.shape[1], data.n_classes, model_type=model_type)

    if log:
        trainer = Trainer(
            deterministic=True,
            max_epochs=max_epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5),
                ModelCheckpoint(
                    monitor="val_loss",
                    dirpath=path,
                    filename=os.path.join(prefix, "checkpoint"),
                    save_top_k=1,
                    verbose=True,
                    mode="min",
                ),
            ],
        )
    else:
        trainer = Trainer(
            max_epochs=max_epochs,
            logger=False,
            enable_checkpointing=False,
            deterministic=True,
        )

    trainer.fit(model, datamodule)
    exp_logger.info(
        f"{prefix} test loss: {trainer.test(model, datamodule, ckpt_path='best')[0]['test_loss']}"
    )
    return load_model(path, prefix)


def load_model(path, prefix="model"):
    return Classifier.load_from_checkpoint(
        os.path.join(path, prefix, "checkpoint.ckpt")
    ).eval()
