"""
Visualization utilities for economic and financial data
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_time_series(data, title="Time Series", figsize=(12, 6), legend=True):
    """
    Plot time series data.
    
    Parameters
    ----------
    data : array-like or DataFrame
        Time series data
    title : str
        Plot title
    figsize : tuple
        Figure size
    legend : bool
        Whether to show legend
    """
    plt.figure(figsize=figsize)
    
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            plt.plot(data.index, data[col], label=col)
    else:
        plt.plot(data)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    if legend:
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data, figsize=(10, 8), cmap='coolwarm'):
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    data : DataFrame or array-like
        Data for correlation calculation
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    """
    if isinstance(data, pd.DataFrame):
        corr = data.corr()
    else:
        corr = np.corrcoef(data.T)
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, 
                square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_residuals(residuals, figsize=(12, 4)):
    """
    Plot diagnostic plots for residuals.
    
    Parameters
    ----------
    residuals : array-like
        Residuals from model
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Time series plot
    axes[0].plot(residuals)
    axes[0].set_title('Residuals Over Time')
    axes[0].set_ylabel('Residual')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_title('Distribution of Residuals')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()


def plot_forecast(actual, forecast, figsize=(12, 6)):
    """
    Plot actual vs forecast values.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    forecast : array-like
        Forecasted values
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.plot(range(len(actual)), actual, 'b-', label='Actual', linewidth=2)
    forecast_range = range(len(actual) - len(forecast), len(actual))
    plt.plot(forecast_range, forecast, 'r--', label='Forecast', linewidth=2)
    
    plt.title('Actual vs Forecast', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
