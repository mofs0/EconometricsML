# EconometricsML: 经济学与金融机器学习综合库

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 项目介绍

**EconometricsML** 是一个面向经济学和商科研究的Python库，集成了：

- **计量经济学模型**: OLS, VAR, GARCH, Panel Data 等经典模型
- **机器学习方法**: 随机森林、梯度提升、神经网络等
- **深度学习应用**: 卷积神经网络、循环神经网络用于时间序列预测
- **强化学习**: 投资组合优化、交易策略学习
- **最新论文模型**: 稀疏VAE、贝叶斯深度学习等

## 核心特性

✅ **开箱即用** - 只需修改数据路径就能直接运行  
✅ **学术标准** - 参考顶级学术期刊和GitHub项目  
✅ **完整文档** - 详细的API文档和中文教程  
✅ **易于扩展** - 模块化设计，便于添加新模型  
✅ **生产级别** - 包含交叉验证、指标评估、可视化  

## 快速开始

### 安装

**方式1: 从源代码安装**

```bash
# 克隆项目
git clone https://github.com/yourusername/econometricsml.git
cd econometricsml

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 安装依赖
pip install -e .

# 安装可选依赖（机器学习）
pip install -e ".[ml]"

# 安装可选依赖（强化学习）
pip install -e ".[rl]"
```

**方式2: 使用pip安装**

```bash
pip install econometricsml
```

### 最小示例

```python
import numpy as np
from econml.econometric_models import OLSRegression

# 生成示例数据
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(100) * 0.5

# 拟合OLS模型
model = OLSRegression()
model.fit(X, y)

# 获取回归摘要
summary = model.summary()
print(f"R²: {summary['r_squared']:.4f}")
print(f"系数: {summary['coefficients']}")

# 预测
y_pred = model.predict(X[:10])
```

## 项目结构

```
econometricsml/
├── econml/
│   ├── __init__.py
│   ├── econometric_models/          # 计量经济学模型
│   │   ├── ols.py                   # 普通最小二乘法
│   │   ├── var.py                   # 向量自回归
│   │   ├── garch.py                 # GARCH波动率模型
│   │   └── __init__.py
│   ├── ml_models/                   # 机器学习模型
│   │   ├── ensemble.py              # 集成模型
│   │   ├── neural_network.py        # 神经网络
│   │   └── __init__.py
│   ├── rl_models/                   # 强化学习模型
│   │   ├── portfolio_rl.py          # 投资组合优化
│   │   └── __init__.py
│   └── utils/                       # 工具函数
│       ├── data_utils.py            # 数据处理
│       ├── visualization.py         # 可视化
│       ├── evaluation.py            # 模型评估
│       └── __init__.py
├── examples/                         # 完整示例代码
│   ├── 01_ols_regression.py
│   ├── 02_var_forecasting.py
│   ├── 03_garch_volatility.py
│   ├── 04_ensemble_prediction.py
│   └── 05_portfolio_optimization.py
├── tutorials/                        # 详细教程
│   ├── 01_introduction.ipynb
│   ├── 02_econometric_models.ipynb
│   ├── 03_machine_learning.ipynb
│   └── 04_reinforcement_learning.ipynb
├── docs/                             # 完整文档
├── tests/                            # 单元测试
├── data/                             # 数据示例
├── README.md                         # 本文件
├── pyproject.toml                    # 项目配置
├── setup.py                          # 安装脚本
└── requirements.txt                  # 依赖清单
```

## 模块详解

### 1. 计量经济学模块 (`econometric_models`)

#### OLS回归

用于经典线性回归和因果推断。

```python
from econml.econometric_models import OLSRegression
import numpy as np

# 准备数据
X = np.random.randn(100, 3)
y = X @ np.array([1, -0.5, 0.3]) + np.random.randn(100) * 0.1

# 拟合模型
model = OLSRegression(fit_intercept=True)
model.fit(X, y)

# 查看统计量
result = model.summary()
print(f"R²: {result['r_squared']}")
print(f"显著性检验 p-值: {result['p_values']}")
```

#### VAR模型 (向量自回归)

用于多变量时间序列预测。

```python
from econml.econometric_models import VARModel

# 多元时间序列数据 (100, 3)
data = np.random.randn(100, 3)

# 拟合VAR(1)
var_model = VARModel(lag_order=1)
var_model.fit(data)

# 预测未来5步
forecast = var_model.forecast(data, steps=5)
```

#### GARCH模型

用于金融时间序列的波动率建模。

```python
from econml.econometric_models import GARCHModel

# 日收益率数据
returns = np.random.randn(200) * 0.02

# 拟合GARCH(1,1)
garch = GARCHModel(p=1, q=1)
garch.fit(returns)

# 预测波动率
volatility_forecast = garch.forecast(returns, steps=5)
```

### 2. 机器学习模块 (`ml_models`)

#### 集成模型

结合多个学习器的预测力。

```python
from econml.ml_models import EnsemblePredictor

# 准备数据
X_train = np.random.randn(200, 10)
y_train = np.random.randn(200)

# 训练集成模型
ensemble = EnsemblePredictor(n_estimators=100)
ensemble.fit(X_train, y_train)

# 预测
predictions = ensemble.predict(X_train[:20])

# 特征重要性
importance = ensemble.feature_importance()
```

#### 神经网络

深度学习回归模型。

```python
from econml.ml_models import NeuralNetworkRegressor

# 创建并训练神经网络
nn = NeuralNetworkRegressor(
    hidden_layers=(128, 64),
    activation='relu',
    learning_rate=0.001
)

nn.fit(X_train, y_train, epochs=200, validation_split=0.2)

# 评估
score = nn.score(X_test, y_test)
predictions = nn.predict(X_test)
```

### 3. 强化学习模块 (`rl_models`)

#### 投资组合优化

使用Q-learning的动态资产配置。

```python
from econml.rl_models import PortfolioRLAgent

# 创建智能体
agent = PortfolioRLAgent(
    n_assets=3,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.1
)

# 训练智能体
# 需要自定义交易环境
# agent.train(env, episodes=1000)

# 获取最优策略
state = np.array([0.5, 0.3, 0.2])  # 当前投资组合
action = agent.policy(state)
```

### 4. 工具函数 (`utils`)

#### 数据加载和预处理

```python
from econml.utils import load_csv_data, prepare_train_test_split, create_time_series_features

# 加载数据
df = load_csv_data('data/economic_data.csv', date_column='Date')

# 创建训练测试集
X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size=0.2)

# 时间序列特征工程
X_ts, y_ts = create_time_series_features(data, lookback=5)
```

#### 可视化

```python
from econml.utils import plot_time_series, plot_correlation_matrix, plot_residuals

# 时间序列绘图
plot_time_series(df[['GDP', 'Inflation', 'Unemployment']], 
                 title='经济主要指标')

# 相关系数矩阵
plot_correlation_matrix(df)

# 残差诊断图
plot_residuals(residuals)
```

#### 模型评估

```python
from econml.utils import calculate_metrics, directional_accuracy, rolling_performance

# 计算指标
metrics = calculate_metrics(y_test, y_pred)
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"MAPE: {metrics['MAPE']:.2f}%")

# 方向准确度（对时间序列预测重要）
dir_acc = directional_accuracy(y_test, y_pred)

# 滚动窗口性能
rolling = rolling_performance(y_test, y_pred, window=20)
```

## 完整示例

### 示例1: GDP增长预测

```python
import numpy as np
import pandas as pd
from econml.econometric_models import OLSRegression
from econml.utils import load_csv_data, prepare_train_test_split, calculate_metrics

# 1. 加载经济数据
data = load_csv_data('data/macro_data.csv')

# 2. 特征准备
X = data[['inflation', 'unemployment', 'interest_rate', 'trade_balance']].values
y = data['gdp_growth'].values

# 3. 分割数据
X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size=0.2)

# 4. 拟合OLS模型
model = OLSRegression()
model.fit(X_train, y_train)

# 5. 预测
y_pred = model.predict(X_test)

# 6. 评估
metrics = calculate_metrics(y_test, y_pred)
print(f"R²: {metrics['R2']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")

# 7. 查看统计显著性
summary = model.summary()
print(f"\n回归系数:\n{summary['coefficients']}")
print(f"\nP值:\n{summary['p_values']}")
```

### 示例2: 股票收益率预测

```python
from econml.ml_models import EnsemblePredictor
from econml.utils import normalize_returns

# 1. 数据准备
prices = load_csv_data('data/stock_prices.csv')['Close'].values

# 2. 转换为对数收益率
returns = normalize_returns(prices)

# 3. 特征工程：创建滞后特征
from econml.utils import create_time_series_features
X, y = create_time_series_features(returns, lookback=5)

X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size=0.2)

# 4. 训练集成模型
ensemble = EnsemblePredictor(n_estimators=100)
ensemble.fit(X_train, y_train)

# 5. 预测
predictions = ensemble.predict(X_test)

# 6. 评估
from econml.utils import directional_accuracy
dir_acc = directional_accuracy(y_test, predictions)
print(f"方向准确度: {dir_acc:.2%}")
```

### 示例3: 波动率预测

```python
from econml.econometric_models import GARCHModel
from econml.utils import calculate_metrics

# 数据准备
returns = # ... 获取日收益率

# 拟合GARCH模型
garch = GARCHModel(p=1, q=1)
garch.fit(returns)

# 历史波动率
historical_vol = garch.conditional_volatility

# 预测未来20天波动率
vol_forecast = garch.forecast(returns, steps=20)

print(f"当前波动率: {historical_vol[-1]:.4f}")
print(f"平均预测波动率: {vol_forecast.mean():.4f}")
```

## 常见用例

### 用例1: 房价预测

```python
# 使用ensemble模型预测房价
from econml.ml_models import EnsemblePredictor

model = EnsemblePredictor()
model.fit(X_features, price_target)  # 面积、位置、房龄等
prices_pred = model.predict(X_new)
```

### 用例2: 需求预测

```python
# 使用VAR进行多变量预测（商品价格、销量、库存等）
from econml.econometric_models import VARModel

var = VARModel(lag_order=2)
var.fit(multivariate_data)  # (T, 3) 其中3个变量：价格、销量、库存
forecast = var.forecast(multivariate_data, steps=6)
```

### 用例3: 风险管理

```python
# 使用GARCH预测波动率，用于VaR计算
from econml.econometric_models import GARCHModel

garch = GARCHModel(p=1, q=1)
garch.fit(returns)
vol = garch.forecast(returns, steps=1)[0]
var_95 = 1.645 * vol  # 95%置信度VaR
```

## 数据格式要求

### CSV文件格式

数据应该是CSV文件，第一行是列名：

```csv
date,gdp,inflation,unemployment
2020-01-01,3.5,2.1,3.8
2020-02-01,3.2,2.2,3.9
...
```

### NumPy数组格式

```python
# 回归：(n_samples, n_features) 特征，(n_samples,) 目标
X = np.random.randn(100, 5)
y = np.random.randn(100)

# 时间序列：(n_samples, n_variables)
ts_data = np.random.randn(200, 3)  # 200个观测，3个变量
```

### Stata联动格式

如果你的数据清洗或实证回归主要在 Stata 中完成，可以用 CSV 或 `.dta` 在 Python 和 Stata 之间往返。

```python
import numpy as np
import pandas as pd

# 1. 在Python中准备或清洗数据
df = pd.DataFrame({
    'firmid': [1, 1, 2, 2],
    'year': [2021, 2022, 2021, 2022],
    'gdp_growth': [3.2, 3.5, 2.8, 3.1],
    'inflation': [2.1, 2.3, 2.0, 2.2],
    'unemployment': [5.0, 4.8, 5.4, 5.1],
})

# 2. 导出给Stata
df.to_csv('data/macro_panel.csv', index=False)
df.to_stata('data/macro_panel.dta', write_index=False)

# 3. 回到Python继续建模
X = df[['inflation', 'unemployment']].values
y = df['gdp_growth'].values
```

```stata
* 4. 在Stata中读取Python导出的数据
use "data/macro_panel.dta", clear
* 或者：import delimited "data/macro_panel.csv", clear varnames(1) encoding(utf8)

* 5. 查看数据
describe
summarize gdp_growth inflation unemployment

* 6. 跑一个基准OLS回归
reg gdp_growth inflation unemployment
estimates store ols_base

* 7. 如果是面板数据，继续做固定效应
xtset firmid year
xtreg gdp_growth inflation unemployment, fe

* 8. 也可以把结果导出为表格继续分析
estimates table ols_base, b(%9.4f) se stats(N r2_a)
```

```python
import pandas as pd

# 9. 将Stata分析后的结果或整理后的数据再导回Python
result_df = pd.read_stata('data/macro_panel.dta')
result_df.to_csv('data/macro_panel_roundtrip.csv', index=False)
```

## 安装可选依赖

### 使用TensorFlow/PyTorch

```bash
# 只安装PyTorch
pip install torch

# 或只安装TensorFlow
pip install tensorflow

# 或一起安装
pip install -e ".[ml]"
```

### 使用完整开发环境

```bash
pip install -e ".[dev]"  # 包括测试和文档工具
```

## 教程和文档

详细教程见 `tutorials/` 目录：

1. **01_introduction.ipynb** - 库的基本使用
2. **02_econometric_models.ipynb** - 计量经济学模型详解
3. **03_machine_learning.ipynb** - 机器学习应用
4. **04_reinforcement_learning.ipynb** - 强化学习在投资中的应用

## 常见问题

### Q: 如何添加自己的数据？

A: 将CSV文件放在 `data/` 目录，然后：
```python
from econml.utils import load_csv_data
df = load_csv_data('data/your_file.csv')
```

### Q: 怎么和 Stata 配合使用？

A: 推荐先在 Python 中清洗数据，再导出 `.dta` 给 Stata：
```python
df.to_stata('data/your_file.dta', write_index=False)
```
在 Stata 中：
```stata
use "data/your_file.dta", clear
reg y x1 x2
```
如果你需要回到 Python 做机器学习评估，也可以再用 `pandas.read_stata()` 读回来。

### Q: 如何扩展库（添加新模型）？

A: 

1. 在相应模块目录创建新Python文件
2. 实现模型类，继承或遵循现有模式
3. 在 `__init__.py` 中导入新类
4. 编写单元测试和示例代码

```python
# 在 econml/econometric_models/ 中创建新文件
class MyNewModel:
    def fit(self, X, y):
        pass
    def predict(self, X):
        pass
```

### Q: 如何处理缺失值？

A: 使用pandas进行数据清理：
```python
df = df.dropna()  # 删除含缺失值的行
df = df.fillna(method='ffill')  # 向前填充
```

### Q: 支持GPU加速吗？

A: 是的。安装PyTorch或TensorFlow后自动支持GPU：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 性能建议

- **小数据集 (<10K行)**: 使用OLS或VAR
- **中数据集 (10K-1M行)**: 使用集成模型或VAR
- **大数据集 (>1M行)**: 使用神经网络或批处理

## 引用

如果在研究中使用本库，请引用：

```bibtex
@software{econometricsml2024,
  title={EconometricsML: A Comprehensive Library for Economic and Financial Machine Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/econometricsml}
}
```

## 贡献指南

欢迎提交Issue和Pull Request！请：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。

## 联系方式

- Issues: [GitHub Issues](https://github.com/yourusername/econometricsml/issues)
- 讨论: [GitHub Discussions](https://github.com/yourusername/econometricsml/discussions)
- 邮箱: your.email@example.com

## 致谢

本项目参考了以下优秀开源项目：

- [statsmodels](https://www.statsmodels.org/)
- [scikit-learn](https://scikit-learn.org/)
- [PyMC](https://www.pymc.io/)
- [Prophet](https://facebook.github.io/prophet/)

---

**最后更新**: 2024年5月  
**版本**: 0.1.0
