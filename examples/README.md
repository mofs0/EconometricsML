# Examples 总览

本目录按主题组织示例和模型说明，按“从简单到复杂”的顺序排列：

- 计量经济学：`examples/econometrics/01_OLS.py` → `10_ProbitLogit.py`
- 机器学习：`examples/ml/01_LinearRegression.py` → `10_ModelEvaluation.py`
- 深度学习：`examples/deep_learning/01_MLP.py` → `05_Autoencoder.py`
- 强化学习：`examples/rl/01_Q_learning.py` → `05_A2C.py`

每个模型文件都包含：

- 中文和 English 的理论说明
- 数学公式（便于直接写进论文或讲义）
- 可直接运行的 Python 代码示例
- 使用该模型的中文论文示例

建议的阅读顺序：先看数字小的文件，再逐步到更复杂的模型。

快速运行示例：

```powershell
python examples/econometrics/01_OLS.py
python examples/ml/01_LinearRegression.py
python examples/deep_learning/01_MLP.py
python examples/rl/01_Q_learning.py
```
# Python 示例索引

这里收集本项目的 Python 示例脚本，全部以 Python 路线为主，不混入 Stata 代码。

## 主题分组

### 计量经济学

- [计量经济学示例](econometrics/README.md)

### 时间序列

- [时间序列示例](timeseries/README.md)

### 机器学习

- [机器学习示例](ml/README.md)

## 当前示例

1. `01_ols_regression.py` - OLS 回归示例
2. `02_var_forecasting.py` - VAR 时间序列预测示例
3. `03_garch_volatility.py` - GARCH 波动率预测示例
4. `04_ensemble_prediction.py` - 集成模型预测示例

## 使用建议

- 想快速上手时，优先从 `01_ols_regression.py` 开始。
- 想看时间序列建模时，继续看 `02_var_forecasting.py` 和 `03_garch_volatility.py`。
- 想做机器学习预测时，重点看 `04_ensemble_prediction.py`。
