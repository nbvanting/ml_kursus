"""Feature engineering functions to ease the data processing."""
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from suncalc import get_position, get_times


def add_day_length(
    dataframe: pd.DataFrame, longitude: float, latitude: float
) -> pd.DataFrame:
    """Adds the day length based on the date, longitude and latitude values."""
    period = pd.date_range(dataframe.index.min(), dataframe.index.max(), freq="D")

    daylengths = []
    for date in period:
        with warnings.catch_warnings():  # catch runtime warning (np.arccos)
            warnings.simplefilter("ignore")
            times = get_times(date, longitude, latitude)

        daylengths.append(
            (times["sunset"] - times["sunrise"]).total_seconds() / 3600
        )  # type: ignore

    daylengths = pd.DataFrame(daylengths, index=period, columns=["day_length"])

    return dataframe.join(daylengths).ffill()


def is_holiday(date: datetime, holidays_arr: dict) -> int:
    """Returns 1 if date is a holiday otherwise returns 0"""
    date = date.replace(hour=0)
    return 1 if (date in holidays_arr) else 0


def add_holiday(dataframe: pd.DataFrame, holidays_arr: dict) -> pd.DataFrame:
    """Adds holidays to the dataframe based on the date and national holidays."""
    return dataframe.assign(
        is_holiday=dataframe.index.to_series().apply(
            lambda x: is_holiday(x, holidays_arr=holidays_arr)
        )
    )


def add_temporal(dataframe: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Adds temporal features based on the datetime index of a DataFrame."""
    temporal_features = {
        "hour": dataframe.index.hour,
        "day_of_week": dataframe.index.day_of_week,
        "week": dataframe.index.isocalendar().week,
        "month": dataframe.index.month,
        "quarter": dataframe.index.quarter,
    }
    for feature in features:
        dataframe[feature] = temporal_features[feature]
    return dataframe


def generate_cyclic_features(
    dataframe: pd.DataFrame,
    col_names: list[str],
    periods: list[int],
    start_nums: list[int],
):
    """Transforms temporal features into 2D cyclic cosine and sine features."""
    for col_name, period, start_num in zip(col_names, periods, start_nums):
        transform = {
            f"sin_{col_name}": lambda x: np.sin(
                2 * np.pi * (dataframe[col_name] - start_num) / period
            ),
            f"cos_{col_name}": lambda x: np.cos(
                2 * np.pi * (dataframe[col_name] - start_num) / period
            ),
        }
        dataframe = dataframe.assign(**transform).drop(columns=[col_name])

    return dataframe


def add_weekends(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Adds a column indicating if the day is a weekend"""
    dataframe = dataframe.assign(weekend=dataframe.index.day_of_week.isin([5, 6]))
    dataframe.weekend = dataframe.weekend.astype(int)
    return dataframe


def add_sun_position(dataframe: pd.DataFrame, lon: float, lat: float) -> pd.DataFrame:
    """Adds the solar position to the dataframe consisting of the Azimuth and Alititude of the sun."""
    sun_position = {
        "sun_azimuth": lambda x: get_position(x.index, lon, lat)["azimuth"],
        "sun_altitude": lambda x: get_position(x.index, lon, lat)["altitude"],
    }

    return dataframe.assign(**sun_position)
