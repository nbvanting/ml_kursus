import multiprocessing as mp
from time import time

import pandas as pd

# PyTorch
import torch
from sklearn.model_selection import train_test_split

# Scikit-Learn
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.prepare_data import TSDataset

features = pd.read_csv(
    "data/electricity_load_data.csv",
    index_col=["datetime"],
    parse_dates=True,
    infer_datetime_format=True,
)

# Generate target and input variables
dataset = TSDataset(dataframe=features, target_variable="load")
X, y = dataset.to_supervised(n_lags=23, horizon=1)

test_ratio = 0.2
val_ratio = test_ratio / (1 - test_ratio)

# Split set once for test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, shuffle=False
)

# Split once more for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=val_ratio, shuffle=False
)

# Reshape to 2D
y_train = y_train.values.reshape(-1, 1)
y_val = y_val.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Scale data
scaler = RobustScaler()

# Scale X
X_train_scl = scaler.fit_transform(X_train)
X_val_scl = scaler.transform(X_val)
X_test_scl = scaler.transform(X_test)

# Scale y
y_train_scl = scaler.fit_transform(y_train)
y_val_scl = scaler.transform(y_val)
y_test_scl = scaler.transform(y_test)


# Create Tensors out of the data
train_features = torch.Tensor(X_train_scl)
train_targets = torch.Tensor(y_train_scl)
val_features = torch.Tensor(X_val_scl)
val_targets = torch.Tensor(y_val_scl)
test_features = torch.Tensor(X_test_scl)
test_targets = torch.Tensor(y_test_scl)

# Torch Tensor Datasets
train_ds = TensorDataset(train_features, train_targets)
val_ds = TensorDataset(val_features, val_targets)
test_ds = TensorDataset(test_features, test_targets)


if __name__ == "__main__":
    for num_workers in range(2, mp.cpu_count() + 1, 2):
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            num_workers=num_workers,
            batch_size=64,
            pin_memory=True,
        )
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
