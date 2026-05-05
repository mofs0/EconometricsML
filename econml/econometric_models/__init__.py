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

__all__ = [
    "OLSRegression",
    "VARModel",
    "GARCHModel",
]
