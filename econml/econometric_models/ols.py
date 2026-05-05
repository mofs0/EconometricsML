"""
Ordinary Least Squares (OLS) Regression

Classical linear regression model with comprehensive diagnostics and statistics.
Suitable for economic regression analysis and causal inference.
"""

import numpy as np
import pandas as pd
from scipy import stats


class OLSRegression:
    """
    Ordinary Least Squares regression model.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to add intercept term
    """
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        self.residuals = None
        self.r_squared = None
        self.adjusted_r_squared = None
        self.std_errors = None
        self.t_stats = None
        self.p_values = None
        self._n_samples = None
        self._n_features = None
        
    def fit(self, X, y):
        """
        Fit the OLS model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target values
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        
        self._n_samples, self._n_features = X.shape
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(self._n_samples), X])
            
        # OLS: beta = (X'X)^-1 X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        
        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficients = beta[1:]
        else:
            self.intercept = 0
            self.coefficients = beta
            
        # Calculate residuals and statistics
        y_pred = self.predict(np.asarray(X)[:, 1:] if self.fit_intercept else X)
        self.residuals = y - y_pred
        
        # R-squared
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)
        
        # Adjusted R-squared
        k = self._n_features + (1 if self.fit_intercept else 0)
        self.adjusted_r_squared = 1 - (1 - self.r_squared) * (self._n_samples - 1) / (self._n_samples - k)
        
        # Standard errors
        mse = ss_res / (self._n_samples - k)
        var_covar = mse * XtX_inv
        self.std_errors = np.sqrt(np.diag(var_covar))
        
        # T-statistics and p-values
        if self.fit_intercept:
            all_coefs = np.concatenate([[self.intercept], self.coefficients])
        else:
            all_coefs = self.coefficients
            
        self.t_stats = all_coefs / self.std_errors
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), self._n_samples - k))
        
        return self
        
    def predict(self, X):
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model must be fit before prediction")
        
        X = np.asarray(X)
        y_pred = X @ self.coefficients + self.intercept
        return y_pred
        
    def summary(self):
        """
        Return a summary of the regression results.
        
        Returns
        -------
        dict
            Dictionary containing regression statistics
        """
        if self.coefficients is None:
            raise ValueError("Model must be fit before summary")
            
        summary = {
            "n_observations": self._n_samples,
            "n_features": self._n_features,
            "r_squared": self.r_squared,
            "adjusted_r_squared": self.adjusted_r_squared,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "std_errors": self.std_errors,
            "t_statistics": self.t_stats,
            "p_values": self.p_values,
        }
        return summary
