import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader

# From src
from src.inference import calculate_metrics

# Horizon (24 hours)
H = 24
TARGET_COLS = [f"load_t+{h}" for h in range(1, H + 1)]
TEST_INDEX = pd.date_range("2021-09-11 14:00:00", "2022-05-14 23:00:00", freq="h")

seed = 65
np.random.seed(seed)
RNG = np.random.RandomState(seed)


@click.command()
@click.option(
    "--kernel",
    default="rbf",
    type=click.STRING,
    help="Kernel type for the algorithm.",
)
@click.option(
    "--c",
    default=1.0,
    type=click.FLOAT,
    help="Regularization parameter.",
)
@click.option(
    "--epsilon",
    default=0.1,
    type=click.FLOAT,
    help="Epsilon of the Support Vector Regression Model.",
)
@click.option(
    "--show_plot",
    default=True,
    type=click.BOOL,
    help="Whether to show a plot of the predictions.",
)
def main(kernel: str, c: int, epsilon: float, show_plot: bool):
    # Convert TensorDatasets back to numpy for the ML algorithms
    train = torch.load("data/train.pt")
    train_loader = DataLoader(train, batch_size=len(train))
    X_train = next(iter(train_loader))[0].numpy()
    y_train = next(iter(train_loader))[1].numpy()

    val = torch.load("data/val.pt")
    val_loader = DataLoader(val, batch_size=len(val))
    X_val = next(iter(val_loader))[0].numpy()
    y_val = next(iter(val_loader))[1].numpy()

    test = torch.load("data/test.pt")
    test_loader = DataLoader(test, batch_size=len(test))
    X_test = next(iter(test_loader))[0].numpy()
    y_test = next(iter(test_loader))[1].numpy()

    scaler = pickle.load(open("data/scaler.pkl", "rb"))

    svr = MultiOutputRegressor(svm.SVR(kernel=kernel, C=c, epsilon=epsilon, verbose=1))
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    predictions = scaler.inverse_transform(y_pred)
    labels = scaler.inverse_transform(y_test)

    df_pred = pd.DataFrame(predictions, columns=TARGET_COLS, index=TEST_INDEX)
    df_true = pd.DataFrame(labels, columns=TARGET_COLS, index=TEST_INDEX)

    calculate_metrics(df_true, df_pred)

    if show_plot:
        plt.figure(figsize=(18, 10))
        plt.plot(df_pred.iloc[0], label="Prediction")
        plt.plot(df_true.iloc[0], label="True")
        plt.legend(fontsize=16)
        plt.xticks(rotation=45)
        plt.title(
            f"Forecast for first 24 hours - MultiOutput Support Vector Regressor {df_true.index[0]}",
            fontsize=18,
        )
        plt.show()


if __name__ == "__main__":
    main()
