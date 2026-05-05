# 快速开始指南

## 5分钟快速开始

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/econometricsml.git
cd econometricsml

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装库
pip install -e .
```

### 2. 最简单的例子

```python
import numpy as np
from econml.econometric_models import OLSRegression

# 生成示例数据
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2 * X[:, 0] - X[:, 1] + np.random.randn(100) * 0.5

# 拟合模型
model = OLSRegression()
model.fit(X, y)

# 预测
predictions = model.predict(X[:10])
print(f"R²: {model.summary()['r_squared']:.4f}")
```

## 常见任务

### 任务1: 加载自己的数据

```python
from econml.utils import load_csv_data

# 假设你的CSV文件格式如下：
# date,variable1,variable2,target
# 2020-01-01,1.5,2.3,3.1
# ...

df = load_csv_data('path/to/your/data.csv')
X = df[['variable1', 'variable2']].values
y = df['target'].values
```

### 任务2: 选择合适的模型

| 问题类型 | 推荐模型 | 代码示例 |
|---------|---------|--------|
| 线性回归/因果推断 | OLSRegression | `from econml.econometric_models import OLSRegression` |
| 多变量时间序列预测 | VARModel | `from econml.econometric_models import VARModel` |
| 波动率预测（金融） | GARCHModel | `from econml.econometric_models import GARCHModel` |
| 非线性预测 | EnsemblePredictor | `from econml.ml_models import EnsemblePredictor` |
| 深度学习 | NeuralNetworkRegressor | `from econml.ml_models import NeuralNetworkRegressor` |

### 任务3: 计算关键指标

```python
from econml.utils import calculate_metrics, directional_accuracy

# 基本指标（RMSE, MAE, R², 等）
metrics = calculate_metrics(y_test, y_pred)

# 对时间序列预测很重要：方向准确度
dir_acc = directional_accuracy(y_test, y_pred)
print(f"预测方向准确度: {dir_acc:.2%}")
```

### 任务4: 可视化结果

```python
from econml.utils import plot_time_series, plot_correlation_matrix, plot_residuals

# 时间序列图
plot_time_series(your_data, title="时间序列")

# 相关系数矩阵
plot_correlation_matrix(your_dataframe)

# 残差诊断图
plot_residuals(model.residuals)
```

## 完整工作流程示例

这是从数据到结果的完整示例：

```python
import numpy as np
import pandas as pd
from econml.econometric_models import OLSRegression
from econml.utils import (
    load_csv_data, 
    prepare_train_test_split, 
    calculate_metrics,
    plot_time_series
)

# 1️⃣ 加载数据
df = load_csv_data('economic_data.csv')

# 2️⃣ 准备特征和目标
X = df[['inflation', 'unemployment', 'interest_rate']].values
y = df['gdp_growth'].values

# 3️⃣ 分割数据
X_train, X_test, y_train, y_test = prepare_train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ 拟合模型
model = OLSRegression(fit_intercept=True)
model.fit(X_train, y_train)

# 5️⃣ 预测
y_pred = model.predict(X_test)

# 6️⃣ 评估
metrics = calculate_metrics(y_test, y_pred)
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"R²: {metrics['R2']:.4f}")

# 7️⃣ 查看统计
summary = model.summary()
print(f"\nR²: {summary['r_squared']:.4f}")
print(f"p值: {summary['p_values']}")

# 8️⃣ 可视化
plot_time_series(df[['gdp_growth']], title='GDP增长')
```

## 运行现有示例

```bash
# 示例1: OLS回归
python examples/01_ols_regression.py

# 示例2: VAR时间序列预测
python examples/02_var_forecasting.py

# 示例3: GARCH波动率预测
python examples/03_garch_volatility.py

# 示例4: 集成模型预测
python examples/04_ensemble_prediction.py
```

## 常见问题

### Q: 导入错误怎么办？

```
ModuleNotFoundError: No module named 'econml'
```

**A**: 确保在项目根目录安装了库：
```bash
pip install -e .
```

### Q: 如何处理缺失值？

```python
import pandas as pd

# 删除含缺失值的行
df = df.dropna()

# 向前填充
df = df.fillna(method='ffill')

# 向后填充
df = df.fillna(method='bfill')
```

### Q: 如何选择训练/测试分割比例？

| 数据量 | 推荐比例 | 说明 |
|--------|---------|------|
| < 1000 | 70/30 | 数据较少，需要更多训练数据 |
| 1000-10000 | 80/20 | 标准比例 |
| > 10000 | 90/10 | 数据充分，可用少量测试数据 |

时间序列数据应该**按时间顺序**分割，不要随机打乱！

```python
# ✅ 正确（时间序列）
train_size = int(0.8 * len(data))
X_train = X[:train_size]
X_test = X[train_size:]

# ❌ 错误（随机分割）
# from sklearn.model_selection import train_test_split
# X_train, X_test, ... = train_test_split(X, y)  # 不要用这个！
```

### Q: 我的模型过拟合怎么办？

1. **增加数据量**
2. **简化模型**（使用更少的特征）
3. **使用正则化**
4. **交叉验证**
5. **增加噪声容限**

### Q: 预测结果不好怎么办？

**检查清单**：

- [ ] 数据质量？检查缺失值、异常值
- [ ] 特征相关性？使用 `plot_correlation_matrix()`
- [ ] 模型选择？尝试不同的模型
- [ ] 超参数？调整学习率、树的深度等
- [ ] 样本量？是否有足够的训练数据
- [ ] 数据泄露？确保测试数据未参与训练
- [ ] 时间序列性质？使用时间序列分割而不是随机分割

## 后续学习

- **详细教程**：查看 `tutorials/` 目录的Jupyter笔记本
- **API文档**：查看 `docs/` 目录
- **更多示例**：查看 `examples/` 目录
- **理论基础**：
  - [OLS回归](https://en.wikipedia.org/wiki/Ordinary_least_squares)
  - [VAR模型](https://en.wikipedia.org/wiki/Vector_autoregression)
  - [GARCH模型](https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity)

## 获取帮助

- 📖 **文档**：[README.md](README.md)
- 💬 **讨论**：[GitHub Discussions](https://github.com/yourusername/econometricsml/discussions)
- 🐛 **报告问题**：[GitHub Issues](https://github.com/yourusername/econometricsml/issues)
- 📧 **邮件**：your.email@example.com

## 下一步

- [ ] 克隆/fork项目
- [ ] 安装依赖
- [ ] 运行示例代码
- [ ] 用自己的数据测试
- [ ] 阅读详细文档
- [ ] 贡献改进建议

---

祝你使用愉快！如有任何问题，欢迎反馈。
