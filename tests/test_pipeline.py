from typing import Union

import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from src.model_training import Pipeline
from src.prepare_data import TSDataset

ScalerType = Union[MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler]


def main() -> None:
    load_data = pd.read_csv(
        "data/load_data.csv", index_col=["datetime"], parse_dates=True
    )
    # Generate target and input variables
    dataset = TSDataset(dataframe=load_data, target_variable="load")
    X, y = dataset.to_supervised(n_lags=23, horizon=1)
    print("With validation")
    print("-" * 60)
    pipe = Pipeline(inputs=X, targets=y, use_validation=True, test_ratio=0.2)
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    train, val, test, y_scaler = pipe.run(x_scaler=x_scaler, y_scaler=y_scaler)
    print("Train:\n", next(iter(train)))
    print("Validation:\n", next(iter(val)))
    print("Test:\n", next(iter(test)))
    print("Without validation")
    print("-" * 60)
    pipe = Pipeline(inputs=X, targets=y, use_validation=False, test_ratio=0.2)
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    train, test, y_scaler = pipe.run(x_scaler=x_scaler, y_scaler=y_scaler)
    print("Train:\n", next(iter(train)))
    print("Test:\n", next(iter(test)))


if __name__ == "__main__":
    main()
