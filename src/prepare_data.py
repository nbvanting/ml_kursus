from typing import Tuple

import pandas as pd


def generate_lags(
    dataframe: pd.DataFrame, n_lags: int, target_variable: str, drop_nan: bool = True) -> pd.DataFrame:
    """Generates n_lags number of lags in dataframe.
    NaN values are removed and columns are added.
    """
    list_of_lags = []
    for lag in range(1, n_lags+1):
        list_of_lags.append(dataframe[target_variable].shift(lag))
    data = pd.concat([dataframe] + list_of_lags, axis=1)
    if drop_nan:
        data = data.iloc[n_lags:]
    data.columns = [target_variable] + [f'{target_variable}-{lag}' for lag in range(1, n_lags+1)]

    return data

def generate_horizons(
    dataframe: pd.DataFrame, horizon: int, target_variable: str, drop_nan: bool = True) -> pd.DataFrame:
    """Generates horizon number of steps in dataframe.
    NaN values are removed.
    """
    data = dataframe.copy()
    for hor in range(1, horizon + 1):
        data[f'{target_variable}+{hor}'] = data[target_variable].shift(-hor)
    if drop_nan:
        data = data.iloc[:-horizon]
    return data


class TSDataset:
    """Class to represent a time series dataset.
    Args:
        dataframe: pd.DataFrame - timeseries dataframe
        target_variable: str - name of the target variable
        drop_nan: bool - whether to drop NaN values created by the transformation
    Methods:
        to_supervised: Transforms the timeseries to a supervised learning dataset
    """
    def __init__(
            self, dataframe: pd.DataFrame, target_variable: str, drop_nan: bool = True) -> None:
        self.dataframe = dataframe
        self.target_variable =  target_variable
        self.drop_nan = drop_nan


    def to_supervised(self, n_lags: int, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transforms timeseries data to a supervised learning problem.
        Args:
            horizon: int - number of steps in the future
            target_variable: str - name of the target column to be transformed
        
        Returns: pandas.DataFrame
            Returns X and y dataframes, where X is the input and y the target variable.
        """

        self.dataframe = generate_lags(
            self.dataframe, n_lags, self.target_variable, self.drop_nan)
        self.dataframe = generate_horizons(
            self.dataframe, horizon, self.target_variable, self.drop_nan)

        X_col = [self.target_variable]\
            + [f'{self.target_variable}-{lag}' for lag in range(1, n_lags+1)]
        y_col = [f'{self.target_variable}+{hor}' for hor in range(1, horizon+1)]

        return self.dataframe[X_col], self.dataframe[y_col]


