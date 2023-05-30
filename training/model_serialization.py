import pickle

import numpy as np
import pandas as pd

# PyTorch Lightning
import pytorch_lightning as pl

# PyTorch
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# From src
from src.model_training import TrainingLoop
from src.models import GatedRecurrentUnit

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
    train, val, test = load_data()
    # Hyperparameters
    input_size = next(iter(train))[0].shape[0]
    output_size = next(iter(train))[1].shape[0]
    hidden_size = 32
    num_layers = 1
    dropout_prob = 0
    learning_rate = 1e-2

    optimizer = "Adam"

    model = GatedRecurrentUnit(
        hidden_size=hidden_size,
        num_rnn_layers=num_layers,
        input_size=input_size,
        output_size=output_size,
        dropout=dropout_prob,
        device=DEVICE,
    )

    train_loop = TrainingLoop(
        model=model,
        datasets=(train, val, test),
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        optimizer=optimizer,
        accelerator=DEVICE,
        train_shuffle=False,
        num_dl_workers=2,
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5)
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models/",
        filename="model-name-{epoch:02d}-{val_loss:.2f}",
        save_weights_only=False,
    )

    trainer = pl.Trainer(
        accelerator=DEVICE,
        devices=1,
        min_epochs=1,
        max_epochs=20,
        gradient_clip_val=0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        callbacks=[early_stop, model_checkpoint],
    )

    trainer.fit(train_loop)
    print("Best model path:", model_checkpoint.best_model_path)


if __name__ == "__main__":
    main()
