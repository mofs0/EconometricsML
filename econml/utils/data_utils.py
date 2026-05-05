"""
Data utilities for loading, preprocessing, and managing economic data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv_data(filepath, date_column=None, index_col=0):
    """
    Load CSV data file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    date_column : str, optional
        Column name to parse as datetime
    index_col : int or str, default=0
        Column(s) to use as index
        
    Returns
    -------
    df : pandas.DataFrame
        Loaded data
    """
    df = pd.read_csv(filepath, index_col=index_col)
    
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
    
    return df


def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Target
    test_size : float, default=0.2
        Proportion of data for test
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    X_train, X_test, y_train, y_test
        Split data
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_time_series_features(data, lookback=1):
    """
    Create lagged features for time series modeling.
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Time series data
    lookback : int, default=1
        Number of lags to create
        
    Returns
    -------
    X : array, shape (n_samples - lookback, n_features * lookback)
        Features with lags
    y : array, shape (n_samples - lookback,)
        Target values
    """
    data = np.asarray(data)
    X, y = [], []
    
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback].flatten())
        y.append(data[i+lookback])
    
    return np.array(X), np.array(y)


def normalize_returns(prices):
    """
    Calculate log returns from price data.
    
    Parameters
    ----------
    prices : array-like
        Price series
        
    Returns
    -------
    returns : array
        Log returns
    """
    prices = np.asarray(prices)
    return np.diff(np.log(prices))
