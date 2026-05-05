# Python 快速开始

这份文档只讲 Python 路线，适合没有 Python 基础、但想先把项目跑起来的同学。

## 1. 安装 Python

Windows 用户建议直接到 [Python 官网](https://www.python.org/downloads/) 下载最新版 Python 3.10+。

安装时注意：

1. 勾选 Add Python to PATH。
2. 保留默认安装即可。
3. 安装完成后打开 PowerShell，输入 `py --version` 检查是否成功。

## 2. 安装项目

```bash
git clone https://github.com/yourusername/econometricsml.git
cd econometricsml

py --version
py -m pip install -U pip
py -m pip install -e .
```

如果你要用机器学习或强化学习相关功能，可以再安装可选依赖：

```bash
py -m pip install -e ".[ml]"
py -m pip install -e ".[rl]"
```

## 3. 推荐 IDE

如果你是新手，优先用下面两个工具：

1. Jupyter Notebook：适合数据清洗、回归、画图和逐步试错。
2. PyCharm：适合完整项目开发、调试和管理多个脚本。

如果你主要是看代码和做笔记，也可以继续用 VS Code，但上手成本通常略高于 Jupyter。

## 4. 启动方式

```bash
py -m pip install notebook jupyterlab
py -m jupyter lab
```

如果你习惯脚本化开发，可以直接在 PyCharm 中打开仓库根目录，并选择刚安装的 Python 解释器。

## 5. 最小示例

```python
import numpy as np
from econml.econometric_models import OLSRegression

np.random.seed(42)
X = np.random.randn(100, 3)
y = 2 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(100) * 0.5

model = OLSRegression()
model.fit(X, y)

print(model.summary())
```
