import pickle

import click
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from torch.utils.data import ConcatDataset

import wandb
from src.model_training import TrainingLoop
from src.models import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "machine-learning-course"


def init_params():
    with open("experiments/configs/config_defaults.yaml", "r", encoding="utf-8") as f:
        try:
            yaml_file = yaml.safe_load(f)
            defaults = dict(yaml_file)
            return defaults
        except yaml.YAMLError as exc:
            print(exc)


def load_data():
    train = torch.load("data/train.pt")
    val = torch.load("data/val.pt")
    test = torch.load("data/test.pt")

    return train, val, test


def tuning():
    wandb.init(config=init_params(), project=PROJECT_NAME, allow_val_change=True)
    params = wandb.config

    train, val, test = load_data()

    # Concatenate train+val
    train = ConcatDataset([train, val])

    input_size = next(iter(train))[0].shape[0]

    model = get_model(params["model"])
    model = model(input_size=input_size, device=DEVICE, **params)
    print(model)

    loop = TrainingLoop(
        model=model,
        datasets=(train, val, test),
        learning_rate=params["lr"],
        batch_size=params["batch_size"],
        optimizer=params["optimizer"],
        accelerator=DEVICE,
        train_shuffle=params["train_dl_shuffle"],
        track_wandb=True,
        num_dl_workers=0,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath="models/",
        filename="final-model-{epoch:02d}",
        save_weights_only=False,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        min_epochs=1,
        max_epochs=params["epochs"],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        callbacks=[model_checkpoint],
    )

    trainer.fit(loop)
    trainer.test(ckpt_path="best")

    predictions, labels = loop.predictions, loop.values
    if DEVICE == "cuda":
        predictions = [tensor.cpu() for tensor in predictions]
        labels = [tensor.cpu() for tensor in labels]

    scaler = pickle.load(open("data/scaler.pkl", "rb"))
    labels = scaler.inverse_transform((np.concatenate(labels, axis=0)))
    predictions = scaler.inverse_transform((np.concatenate(predictions, axis=0)))

    mae = mean_absolute_error(labels, predictions)
    rmse = mean_squared_error(labels, predictions) ** 0.5
    r2 = r2_score(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions) * 100

    wandb.log({"pred_mae": mae, "pred_rmse": rmse, "pred_r2": r2, "pred_mape": mape})


@click.command()
@click.option(
    "--yaml_config",
    default="final_config",
    type=click.STRING,
    help="Filename of sweep config (excl. file extension)",
)
@click.option(
    "--count", default=3, type=click.INT, help="Number of models to run in the sweep."
)
def main(yaml_config: str, count: int):
    with open(f"experiments/configs/{yaml_config}.yaml", "r", encoding="utf-8") as f:
        file = yaml.safe_load(f)
        sweep_config = dict(file)

    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)

    wandb.agent(sweep_id, function=tuning, count=count)

    wandb.finish()


if __name__ == "__main__":
    main()
