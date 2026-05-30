"""empirlab.ml —— 机器学习计量方法子包。

模块列表
--------
double_ml        : Double/Debiased Machine Learning（Chernozhukov et al. 2018）
lasso_select     : LASSO 变量筛选（Tibshirani 1996 + Belloni et al. 2014）
random_forest    : 随机森林回归 / 分类（Breiman 2001）
causal_forest    : 因果森林（Wager & Athey 2018）
"""

from .double_ml import DoubleML, dml_data
from .lasso_select import LassoSelect, lasso_data
from .random_forest import RFRegressor, rf_data
from .causal_forest import CausalForest, cf_data

__all__ = [
    "DoubleML", "dml_data",
    "LassoSelect", "lasso_data",
    "RFRegressor", "rf_data",
    "CausalForest", "cf_data",
]
