"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Model

For modeling volatility in financial time series, essential for risk management
and derivative pricing in finance.
"""

import numpy as np
from scipy.optimize import minimize
import warnings


class GARCHModel:
    """
    GARCH(p, q) model for conditional volatility.
    
    Parameters
    ----------
    p : int, default=1
        Order of ARCH term
    q : int, default=1
        Order of GARCH term
    """
    
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.params = None
        self.residuals = None
        self.conditional_volatility = None
        
    def _likelihood(self, params, returns):
        """Calculate negative log-likelihood."""
        omega, alpha_params, beta_params = (
            params[0],
            params[1:self.p+1],
            params[self.p+1:self.p+self.q+1]
        )
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            arch_term = np.sum(alpha_params * returns[max(0, t-self.p):t]**2)
            garch_term = np.sum(beta_params * sigma2[max(0, t-self.q):t])
            sigma2[t] = omega + arch_term + garch_term
            
            # Ensure positive variance
            if sigma2[t] <= 0:
                return 1e10
        
        # Log-likelihood
        ll = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
        return -ll
        
    def fit(self, returns):
        """
        Fit the GARCH model.
        
        Parameters
        ----------
        returns : array-like, shape (n_samples,)
            Time series of returns (assumed to have zero mean)
            
        Returns
        -------
        self
        """
        returns = np.asarray(returns)
        
        # Initial parameters
        x0 = np.concatenate([
            [0.01],  # omega
            [0.1] * self.p,  # alpha
            [0.8] * self.q,  # beta
        ])
        
        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._likelihood,
                x0,
                args=(returns,),
                method='BFGS'
            )
        
        self.params = result.x
        
        # Calculate conditional volatility
        self._update_conditional_volatility(returns)
        
        return self
        
    def _update_conditional_volatility(self, returns):
        """Update conditional volatility."""
        omega = self.params[0]
        alpha_params = self.params[1:self.p+1]
        beta_params = self.params[self.p+1:self.p+self.q+1]
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            arch_term = np.sum(alpha_params * returns[max(0, t-self.p):t]**2)
            garch_term = np.sum(beta_params * sigma2[max(0, t-self.q):t])
            sigma2[t] = omega + arch_term + garch_term
        
        self.conditional_volatility = np.sqrt(sigma2)
        self.residuals = returns / self.conditional_volatility
        
    def forecast(self, returns, steps=1):
        """
        Forecast volatility.
        
        Parameters
        ----------
        returns : array-like
            Historical returns
        steps : int
            Number of steps ahead
            
        Returns
        -------
        forecast : array
            Forecasted volatility
        """
        if self.params is None:
            raise ValueError("Model must be fit before forecasting")
        
        returns = np.asarray(returns)
        self._update_conditional_volatility(returns)
        
        omega = self.params[0]
        alpha_params = self.params[1:self.p+1]
        beta_params = self.params[self.p+1:self.p+self.q+1]
        
        forecast = np.zeros(steps)
        sigma2_prev = self.conditional_volatility[-1]**2
        
        for h in range(steps):
            sigma2_h = omega + np.sum(alpha_params) * returns[-1]**2 + np.sum(beta_params) * sigma2_prev
            forecast[h] = np.sqrt(sigma2_h)
            sigma2_prev = sigma2_h
        
        return forecast
