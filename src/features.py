"""
Feature engineering functions for time-series energy forecasting.
"""


def add_time_features(df, time_col="time"):
    """Extract time-based features from datetime column."""
    df = df.copy()
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["quarter"] = df[time_col].dt.quarter
    df["year"] = df[time_col].dt.year
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["day_of_year"] = df[time_col].dt.dayofyear
    return df


def add_lag_features(df, target_col, lags):
    """
    Add lag features for the target variable.

    Args:
        df: DataFrame sorted by time
        target_col: name of the target column
        lags: list of lag values, e.g. [1, 2, 24, 48, 168]
              1 = previous hour
              24 = same hour yesterday
              168 = same hour last week
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col, windows):
    """
    Add rolling mean and std features.

    Args:
        df: DataFrame sorted by time
        target_col: name of the target column
        windows: list of window sizes, e.g. [6, 12, 24, 168]
    """
    df = df.copy()
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = (
            df[target_col].shift(1).rolling(window=window).mean()
        )
        df[f"{target_col}_rolling_std_{window}"] = (
            df[target_col].shift(1).rolling(window=window).std()
        )
    return df


def create_features(df, target_col="total load actual", time_col="time"):
    """
    Full feature engineering pipeline.
    Call this to create all features at once.
    """
    df = add_time_features(df, time_col)

    df = add_lag_features(df, target_col, lags=[1, 2, 3, 24, 48, 168])

    df = add_rolling_features(df, target_col, windows=[6, 12, 24, 168])

    return df
