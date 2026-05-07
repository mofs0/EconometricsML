"""`empirlab.traditional` 包初始化。

这里集中导出传统计量模型，方便在 notebook 或脚本中统一导入。
"""

from .ols import OLS
from .iv import IV2SLS

__all__ = ["OLS", "IV2SLS"]
"""Traditional econometric models."""

from empirlab.traditional.ols import OLS

__all__ = ["OLS"]
