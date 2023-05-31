"""Contains training loop classes and helper functions for the training of neural networks."""
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, RobustScaler,
                                   StandardScaler)
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ScalerType = Union[MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler]
ActivationType = Union[
    nn.ELU,
    nn.Hardshrink,
    nn.Hardswish,
    nn.LeakyReLU,
    nn.PReLU,
    nn.ReLU,
    nn.RReLU,
    nn.SELU,
    nn.CELU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
]
OptimizerType = Union[
    optim.Adadelta,
    optim.Adagrad,
    optim.Adam,
    optim.AdamW,
    optim.Adamax,
    optim.ASGD,
    optim.LBFGS,
    optim.NAdam,
    optim.RAdam,
    optim.RMSprop,
    optim.Rprop,
    optim.SGD,
]


def get_activation(activation: str) -> ActivationType:
    """
    Function to select an activation function from the available PyTorch algorithms.
    https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
    """

    activations = {
        "elu": nn.ELU,
        "hardshrink": nn.Hardshrink,
        "hardswish": nn.Hardswish,
        "leakyrelu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "relu": nn.ReLU,
        "rrelu": nn.RReLU,
        "selu": nn.SELU,
        "celu": nn.CELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
    }
    return activations.get(activation.lower())  # type: ignore


def get_optimizer(optimizer: str) -> OptimizerType:
    """
    Function to select an optimizer from the available PyTorch optimizers.
    https://pytorch.org/docs/stable/optim.html#algorithms"""

    optimizers = {
        "adadelta": optim.Adadelta,
        "adagrad": optim.Adagrad,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "lbfgs": optim.LBFGS,
        "nadam": optim.NAdam,
        "radam": optim.RAdam,
        "rmsprop": optim.RMSprop,
        "rprop": optim.Rprop,
        "sgd": optim.SGD,
    }

    return optimizers.get(optimizer.lower())  # type: ignore


def get_scaler(scaler: str) -> ScalerType:
    """Function to select and switch between scalers.
    Args:
        scaler (str): Name of a sklearn scaler
            (MinMaxScaler(), StandardScaler(), MaxAbsScaler(), RobustScaler())
    """
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())  # type: ignore


class TrainingLoop(LightningModule):
    """A training loop implementation using PyTorch Lightning."""

    def __init__(
        self,
        model: nn.Module,
        datasets: tuple,
        learning_rate: float,
        batch_size: int,
        optimizer: str,
        accelerator: str,
        train_shuffle: bool = True,
        num_dl_workers: int = 0,
        track_wandb: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = model

        # Training parameters
        self.datasets = datasets
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = get_optimizer(optimizer)
        self.accelerator = accelerator
        self.train_shuffle = train_shuffle
        self.num_dl_workers = num_dl_workers
        self.track_wandb = track_wandb

        # Predictions
        self.predictions: List[float] = []
        self.values: List[float] = []

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view([x.size(0), -1, self.model.input_size]).to(self.accelerator)
        y = y.to(self.accelerator)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        if self.track_wandb:
            wandb.log({"train_loss": loss, "train_mae": mae})
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view([x.size(0), -1, self.model.input_size]).to(self.accelerator)
        y = y.to(self.accelerator)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log("val_loss", loss)
        if self.track_wandb:
            wandb.log({"val_loss": loss, "val_mae": mae})
        return {"val_loss": loss, "log": {"val_loss": loss}}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view([x.size(0), -1, self.model.input_size]).to(self.accelerator)
        y = y.to(self.accelerator)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        if self.track_wandb:
            wandb.log({"test_loss": loss, "test_mae": mae})
        self.predictions.append(y_hat)
        self.values.append(y)
        return {"test_loss": loss, "log": {"test_loss": loss}}

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o["val_loss"] for o in outputs]) / len(outputs)

        results = {
            "progress_bar": {"val_loss": val_loss_mean.item()},
            "log": {"val_loss": val_loss_mean.item()},
        }
        return results

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)  # type: ignore
        return optimizer

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.ts_train, self.ts_val = self.datasets[0], self.datasets[1]

        if stage == "test" or stage is None:
            self.ts_test = self.datasets[2]

    def train_dataloader(self):
        return DataLoader(
            self.datasets[0],
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            drop_last=True,
            num_workers=self.num_dl_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[1],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_dl_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[2],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_dl_workers,
        )


class Pipeline:
    """A useful pipeline class for preparing data for neural network training.
    The pipeline splits, reshapes, scales, and creates tensors of the input data.
    """

    def __init__(
        self,
        inputs: pd.DataFrame,
        targets: pd.DataFrame,
        use_validation: bool = True,
        test_ratio: float = 0.2,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.use_validation = use_validation
        self.test_ratio = test_ratio

    def _split_data(self) -> Tuple[Any, Any, Optional[Any], Optional[Any], Any, Any]:
        x_train, x_test, y_train, y_test = train_test_split(
            self.inputs, self.targets, test_size=self.test_ratio, shuffle=False
        )

        if self.use_validation:
            val_ratio = self.test_ratio / (1 - self.test_ratio)
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=val_ratio, shuffle=False
            )
            return x_train, y_train, x_val, y_val, x_test, y_test
        else:
            return x_train, y_train, None, None, x_test, y_test

    def _reshape(self, y_set: Any) -> Any:
        return y_set.values.reshape(-1, y_set.values.shape[1])

    def _scale(
        self,
        x_train: Any,
        y_train: Any,
        x_val: Optional[Any],
        y_val: Optional[Any],
        x_test: Any,
        y_test: Any,
        x_scaler: ScalerType,
        y_scaler: ScalerType,
    ):
        x_train_scaled = Tensor(x_scaler.fit_transform(x_train))
        y_train_scaled = Tensor(y_scaler.fit_transform(y_train))
        x_val_scaled = (
            Tensor(x_scaler.transform(x_val)) if self.use_validation else None
        )
        y_val_scaled = (
            Tensor(y_scaler.transform(y_val)) if self.use_validation else None
        )
        x_test_scaled = Tensor(x_scaler.transform(x_test))
        y_test_scaled = Tensor(y_scaler.transform(y_test))

        return (
            x_train_scaled,
            y_train_scaled,
            x_val_scaled,
            y_val_scaled,
            x_test_scaled,
            y_test_scaled,
            y_scaler,
        )

    def _make_tensor_dataset(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_val: Any,
        y_val: Any,
        x_test: Tensor,
        y_test: Tensor,
    ) -> Tuple[TensorDataset, Optional[TensorDataset], TensorDataset]:
        train_ds = TensorDataset(x_train, y_train)
        val_ds = (
            TensorDataset(Tensor(x_val), Tensor(y_val)) if self.use_validation else None
        )
        test_ds = TensorDataset(x_test, y_test)
        return train_ds, val_ds, test_ds

    def run(self, x_scaler: ScalerType, y_scaler: ScalerType):
        x_train, y_train, x_val, y_val, x_test, y_test = self._split_data()
        y_train = self._reshape(y_train)
        y_val = self._reshape(y_val) if self.use_validation else None
        y_test = self._reshape(y_test)
        (
            x_train_scaled,
            y_train_scaled,
            x_val_scaled,
            y_val_scaled,
            x_test_scaled,
            y_test_scaled,
            scaler,
        ) = self._scale(
            x_train, y_train, x_val, y_val, x_test, y_test, x_scaler, y_scaler
        )
        train_ds, val_ds, test_ds = self._make_tensor_dataset(
            x_train_scaled,
            y_train_scaled,
            x_val_scaled,
            y_val_scaled,
            x_test_scaled,
            y_test_scaled,
        )
        if self.use_validation:
            return train_ds, val_ds, test_ds, scaler
        return train_ds, test_ds, y_scaler
