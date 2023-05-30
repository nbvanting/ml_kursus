import pickle

import numpy as np
import pandas as pd

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Scikit-Learn
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# From src
from src.model_training import TrainingLoop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 32


def load_data():
    train = torch.load("data/train.pt")
    val = torch.load("data/val.pt")
    test = torch.load("data/test.pt")
    return train, val, test


def inference(predictions, values):
    if DEVICE == "cuda":
        predictions = [tensor.cpu() for tensor in predictions]
        values = [tensor.cpu() for tensor in values]

    preds = np.concatenate(predictions, axis=0)
    vals = np.concatenate(values, axis=0)

    scaler = pickle.load(open("data/scaler.pkl", "rb"))

    preds = scaler.inverse_transform(preds)
    vals = scaler.inverse_transform(vals)
    return preds, vals


def main():
    best_training_path = "models/model-name-epoch=06-val_loss=0.49.ckpt"

    train_loop = TrainingLoop.load_from_checkpoint(best_training_path)

    early_stop = EarlyStopping(monitor="val_loss", patience=1)
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models/",
        filename="model-name-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(
        accelerator=DEVICE,
        devices=1,
        min_epochs=0,
        max_epochs=20,
        callbacks=[early_stop, model_checkpoint],
    )

    trainer.fit(train_loop, ckpt_path=best_training_path)
    trainer.test(ckpt_path=best_training_path)

    predictions, targets = inference(train_loop.predictions, train_loop.values)

    target_cols = [f"load+{h}" for h in range(1, 25)]
    test_index = pd.date_range("2021-09-11 14:00:00", "2022-05-14 23:00:00", freq="h")

    df_preds = pd.DataFrame(
        predictions, columns=target_cols, index=test_index[: len(targets)]
    )
    df_vals = pd.DataFrame(
        targets, columns=target_cols, index=test_index[: len(targets)]
    )

    print("TEST PERFORMANCE")
    print("RMSE:\t", mean_squared_error(df_vals, df_preds) ** 0.5)
    print("MAE:\t", mean_absolute_error(df_vals, df_preds))
    print("MAPE:\t", mean_absolute_percentage_error(df_vals, df_preds) * 100)
    print("R^2:\t", r2_score(df_vals, df_preds))


if __name__ == "__main__":
    main()
