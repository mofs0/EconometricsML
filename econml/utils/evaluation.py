"""
Model evaluation metrics and utilities
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Diebold-Mariano statistic (for time series)
    residuals = y_true - y_pred
    dm_stat = np.mean(residuals**2)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'DM': dm_stat,
    }


def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy for time series predictions.
    
    Measures the proportion of times the predicted direction
    (up/down) matches the actual direction.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    accuracy : float
        Proportion of correct direction predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    
    accuracy = np.mean(true_dir == pred_dir)
    return accuracy


def rolling_performance(y_true, y_pred, window=20):
    """
    Calculate rolling window performance metrics.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    window : int
        Rolling window size
        
    Returns
    -------
    rolling_metrics : dict
        Dictionary with rolling metrics
    """
    n = len(y_true) - window + 1
    rolling_mse = np.zeros(n)
    rolling_mae = np.zeros(n)
    
    for i in range(n):
        rolling_mse[i] = mean_squared_error(y_true[i:i+window], y_pred[i:i+window])
        rolling_mae[i] = mean_absolute_error(y_true[i:i+window], y_pred[i:i+window])
    
    return {
        'rolling_mse': rolling_mse,
        'rolling_mae': rolling_mae,
        'mean_mse': np.mean(rolling_mse),
        'mean_mae': np.mean(rolling_mae),
    }
