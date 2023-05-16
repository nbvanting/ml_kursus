"""This module contains utility functions to ease the inference process of a trained deep learning model"""

from typing import Any, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

ScalerType = Union[MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler]


def format_predictions(
    predictions: List[Any],
    labels: List[Any],
    x_test: pd.DataFrame,
    scaler: ScalerType,
    to_cpu: bool = False,
) -> pd.DataFrame:
    """Function to format predictions from numpy arrays to a single pandas DataFrame.
    Values are moved to CPU device and reshaped.
    Args:
        predictions: list[np.ndarray] A list of predictions for each batch.
        labels: list[np.ndarray] A list of labels for each batch.
        X_test: pd.DataFrame The unscaled test data set of the input variables.
        scaler: ScalerType Union[MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler]
        to_cpu: bool Flag to indicate if values should be moved to CPU device
    """
    if to_cpu:
        labels = [tensor.cpu() for tensor in labels]
        predictions = [tensor.cpu() for tensor in predictions]

    labels = scaler.inverse_transform((np.concatenate(labels, axis=0)))[:, 0]
    predictions = scaler.inverse_transform((np.concatenate(predictions, axis=0)))[:, 0]

    results = pd.DataFrame(
        data={"labels": labels, "prediction": predictions},
        index=x_test.head(len(labels)).index,
    )
    return results.sort_index()


def symmetric_mape(
    labels: Union[pd.DataFrame, pd.Series], predictions: Union[pd.DataFrame, pd.Series]
) -> np.float64:
    """Returns the symmetric mean absolute percentage error of labels and predicted values."""
    return (
        100
        / len(labels)
        * np.sum(
            2 * np.abs(predictions - labels) / (np.abs(labels) + np.abs(predictions))
        )
    )


def calculate_metrics(
    labels: Union[pd.DataFrame, pd.Series], predictions: Union[pd.DataFrame, pd.Series]
) -> dict:
    """Function to calculate metrics of predicted values of a model given the labels values.
    Args:
        labels: The label values
        prediction: The predicted values
    Returns:
        Prints all metrics and returns a dictionary of the metrics.
    """
    results = {
        "mae": mean_absolute_error(labels, predictions),
        "mape": mean_absolute_percentage_error(labels, predictions) * 100,
        "smape": symmetric_mape(labels, predictions),
        "rmse": mean_squared_error(labels, predictions) ** 0.5,
        "r2": r2_score(labels, predictions),
    }
    print("Mean Absolute Error:", results["mae"])
    print("Mean Absolute % Error:", results["mape"])
    print("Symmetric Mean Absolute % Error:", results["mape"])
    print("Root Mean Squared Error:", results["rmse"])
    print("R^2 Score:", results["r2"])

    return results
