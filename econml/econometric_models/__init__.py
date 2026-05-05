"""
Econometric Models Module

Contains classical econometric models including:
- Ordinary Least Squares (OLS)
- Vector Autoregression (VAR)
- GARCH models
- Panel data models
- Time series models
"""

from .ols import OLSRegression
from .var import VARModel
from .garch import GARCHModel
from .extended import (
    DifferenceInDifferences,
    FixedEffectsRegression,
    IVRegression,
    LogitRegression,
    ProbitRegression,
    RandomEffectsRegression,
)

__all__ = [
    "OLSRegression",
    "VARModel",
    "GARCHModel",
    "IVRegression",
    "FixedEffectsRegression",
    "RandomEffectsRegression",
    "DifferenceInDifferences",
    "LogitRegression",
    "ProbitRegression",
]
