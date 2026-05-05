"""
Vector Autoregression (VAR) Model

For multivariate time series analysis in economic systems.
Useful for studying relationships between multiple economic variables.
"""

import numpy as np
import pandas as pd
from scipy import stats


class VARModel:
    """
    Vector Autoregression (VAR) model for multivariate time series.
    
    Parameters
    ----------
    lag_order : int, default=1
        Number of lags to include
    """
    
    def __init__(self, lag_order=1):
        self.lag_order = lag_order
        self.coefficients = None
        self.covariance = None
        self._n_vars = None
        
    def _create_lagged_data(self, data):
        """Create lagged version of data for VAR estimation."""
        n_obs, n_vars = data.shape
        
        # Create matrix of lagged values
        X_list = [np.ones(n_obs - self.lag_order)]
        for lag in range(1, self.lag_order + 1):
            X_list.append(data[self.lag_order - lag:-lag if lag > 0 else None])
        
        X = np.column_stack(X_list)
        y = data[self.lag_order:]
        
        return X, y
        
    def fit(self, data):
        """
        Fit the VAR model.
        
        Parameters
        ----------
        data : array-like, shape (n_samples, n_variables)
            Multivariate time series data
            
        Returns
        -------
        self
        """
        data = np.asarray(data)
        self._n_vars = data.shape[1]
        
        X, y = self._create_lagged_data(data)
        
        # Estimate coefficients using OLS
        # y = X * beta, so beta = (X'X)^-1 X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        
        self.coefficients = beta
        
        # Calculate residuals and covariance
        y_pred = X @ beta
        residuals = y - y_pred
        self.covariance = (residuals.T @ residuals) / (len(residuals) - self.lag_order - 1)
        
        return self
        
    def forecast(self, data, steps=1):
        """
        Generate forecasts from the fitted VAR model.
        
        Parameters
        ----------
        data : array-like, shape (n_samples, n_variables)
            Historical data for initialization
        steps : int, default=1
            Number of steps ahead to forecast
            
        Returns
        -------
        forecast : array, shape (steps, n_variables)
            Forecasted values
        """
        if self.coefficients is None:
            raise ValueError("Model must be fit before forecasting")
            
        data = np.asarray(data)
        forecast = np.zeros((steps, self._n_vars))
        current_data = data.copy()
        
        for step in range(steps):
            # Construct lagged feature matrix
            X_pred = np.ones(1)
            for lag in range(1, self.lag_order + 1):
                X_pred = np.concatenate([X_pred, current_data[-lag]])
            
            # Predict next values
            next_vals = X_pred @ self.coefficients
            forecast[step] = next_vals
            
            # Update current data with prediction
            current_data = np.vstack([current_data, next_vals])
            
        return forecast
