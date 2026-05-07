# GitHub Copilot 代码生成规则 — EmpirLab

> **这是项目的最高优先级约束文件。**  
> Copilot、Claude、或任何 AI 助手在此仓库中生成代码时，必须严格遵守以下所有规则。  
> **违反规则宁可停止生成，也不能擅自修改项目结构。**

---

## 零、当前项目范围（必读）

**当前阶段只开发经济学 · 金融学 · 商科方向。**

其他学科（医学统计、社会学等）架构已预留但**尚未开发**，
AI 助手不得擅自在 `empirlab/bio/`、`empirlab/social/` 等目录下生成代码。

**近期优先级（按顺序）：**
1. `traditional/` 补全：`logit.py` → `var.py` 完整实现
2. 每新增一个 Python 模型，同步新增对应 Stata `.do` 文件
3. Notebook 补全：`T02_IV`、`T03_DiD`、`T04_Panel`
4. `dl/` 实现：`lstm.py`、`transformer_ts.py`
5. `rl/` 实现：`dqn.py`、`ppo.py`

---

## 一、项目架构规则（最重要，不得违反）

### 1.1 目录与文件的职责是固定的，不得创建规则之外的文件

```
empirlab/traditional/   → 只放传统计量统计模型的 Python 类
empirlab/ml/            → 只放机器学习模型的 Python 类
empirlab/dl/            → 只放深度学习模型的 Python 类（统一用 PyTorch）
empirlab/rl/            → 只放强化学习模型的 Python 类（统一用 Gymnasium 接口）
empirlab/llm/           → 只放 LLM 调用逻辑（不是训练，是使用）
empirlab/utils/         → 只放工具函数（data_io, preprocessing, visualization, metrics）
stata/traditional/      → 只放与 empirlab/traditional/ 对应的 .do 文件
stata/ml/               → 只放与 empirlab/ml/ 对应的 .do 文件
notebooks/              → 只放 .ipynb 文件，不放 .py 文件
gotools/                → 只放 Go 代码，不放 Python 代码
```

❌ **禁止行为：**

- 在 `empirlab/` 下直接放 notebook 或示例脚本
- 在 `notebooks/` 下放 `.py` 文件
- 创建 `scripts/models/` 这类不在规则内的目录
- 把多个模型合并到一个文件里（如 `all_models.py`）
- 把 `utils/` 的内容挪到模型文件里内联实现

---

### 1.2 每个模型文件必须是独立的

每个 `empirlab/*/xxx.py` 文件：

- ✅ 只允许 import `empirlab.utils`（内部工具）和标准第三方库
- ✅ 必须包含一个主类，类名与文件名对应（如 `ols.py` → `class OLS`）
- ✅ 文件底部必须有 `if __name__ == "__main__":` 最小可运行示例
- ❌ 不允许 import 同级其他模型文件（如在 `iv.py` 里 import `ols.py`）
- ❌ 不允许定义超过 3 个以上不相关的类在同一个文件（除非是同一模型族，如 Logit/Probit 共一个文件）

---

### 1.3 深度学习框架：统一 PyTorch，不使用 TensorFlow / Keras

```python
# ✅ 正确
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ❌ 禁止
import tensorflow as tf
from keras.models import Sequential
```

---

### 1.4 强化学习：统一 Gymnasium 接口

```python
# ✅ 正确
import gymnasium as gym
from gymnasium import spaces

# ❌ 禁止使用旧版 gym（已弃用）
import gym
```

---

## 二、Python 模型类的标准模板

所有 `empirlab/` 下的模型类必须遵循以下结构，**缺少任何部分都不完整**：

```python
"""
模型名称 (英文缩写)
===================
来源参考：[论文/教材名称]
适用场景：[一句话说明]
Python 版本：3.10+
依赖：statsmodels >= 0.14 / scikit-learn >= 1.3 / torch >= 2.0（按需填写）
"""

import numpy as np
import pandas as pd
# ... 其他必要导入

from empirlab.utils.metrics import calculate_metrics  # 只从 utils 引用


class ModelName:
    """
    模型全称 (英文名)

    参数
    ----
    param1 : type
        参数说明
    param2 : type, optional
        参数说明，默认值为 xxx

    示例
    ----
    >>> model = ModelName(param1=value)
    >>> model.fit(X, y)
    >>> result = model.summary()
    """

    def __init__(self, param1, param2=default_value):
        self.param1 = param1
        self.param2 = param2
        self.is_fitted = False
        # 初始化其他属性

    def fit(self, X, y, **kwargs):
        """
        拟合模型。

        参数
        ----
        X : np.ndarray or pd.DataFrame, shape (n_samples, n_features)
            特征矩阵
        y : np.ndarray or pd.Series, shape (n_samples,)
            目标变量
        """
        # 实现
        self.is_fitted = True
        return self

    def predict(self, X):
        """预测。"""
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 方法。")
        # 实现
        pass

    def summary(self):
        """返回模型统计摘要（字典格式）。"""
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 方法。")
        # 实现，返回 dict
        pass


# ============================================================
# 最小运行示例（必须可以直接 python xxx.py 运行，无需额外数据）
# ============================================================
if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    beta = np.array([1.5, -0.8, 0.3])
    y = X @ beta + np.random.randn(n) * 0.5

    model = ModelName()
    model.fit(X, y)
    result = model.summary()
    print("模型摘要：", result)
```

---

## 三、Stata `.do` 文件的标准模板

对应每个有 Stata 版本的 Python 模型，`.do` 文件必须包含：

```stata
/*==============================================================
  文件名：XX_model_name.do
  模型：  模型全称
  描述：  一句话说明
  依赖：  需要的外部命令（ssc install xxx）
  数据：  使用的数据集
  作者：  （留空，由使用者填写）
  更新：  2025-05
==============================================================*/

* ---------------------------------------------------------------
* 0. 全局设置（使用者修改这里的路径）
* ---------------------------------------------------------------
global root  "C:/Users/yourname/empirlab"    // ← 修改为你的路径
global data  "$root/data/sample"
global out   "$root/output"

clear all
set more off
cd "$root"

* ---------------------------------------------------------------
* 1. 加载数据
* ---------------------------------------------------------------
use "$data/firm_panel.dta", clear
// 或者
// import delimited "$data/macro_china.csv", clear

* ---------------------------------------------------------------
* 2. 数据预处理
* ---------------------------------------------------------------
* （变量生成、缩尾处理等）
winsor2 var1 var2, replace cuts(1 99)

* ---------------------------------------------------------------
* 3. 核心估计
* ---------------------------------------------------------------
* （模型命令）

* ---------------------------------------------------------------
* 4. 结果输出
* ---------------------------------------------------------------
* 建议用 outreg2 或 esttab 导出表格
// esttab using "$out/table1.csv", replace star(* 0.1 ** 0.05 *** 0.01)

* ---------------------------------------------------------------
* 5. 稳健性检验
* ---------------------------------------------------------------
```

---

## 四、Notebook `.ipynb` 的结构规则

每个 notebook 必须包含以下**六段 Markdown 标题**，顺序不得调换：

```
## 0. 论文信息
## 1. 研究设计与识别策略
## 2. 数学理论与模型
## 3. 数据加载与预处理
## 4. 模型估计
## 5. 结果解读与稳健性检验
## 6. 可视化
```

**Notebook 内引用模型的方式（不得在 notebook 里重写模型代码）：**

```python
# ✅ 正确：从 empirlab 包导入
from empirlab.traditional import OLS
from empirlab.ml import DoubleML

# ❌ 禁止：在 notebook 里内联定义模型类
class OLS:
    def fit(self): ...
```

**数学公式使用 LaTeX，必须能在 Jupyter 中正确渲染：**

```markdown
$$
\hat{\beta}_{OLS} = (X^{\top}X)^{-1}X^{\top}y
$$
```

**提交前必须 Kernel → Restart & Run All，确保所有 cell 有输出。**

---

## 五、命名规范

### Python 文件和类

| 对象     | 规范            | 示例                              |
| -------- | --------------- | --------------------------------- |
| 模块文件 | `snake_case.py` | `causal_forest.py`                |
| 类名     | `PascalCase`    | `CausalForest`                    |
| 方法名   | `snake_case`    | `fit()`, `predict()`, `summary()` |
| 内部变量 | `snake_case`    | `n_samples`, `beta_hat`           |
| 常量     | `UPPER_SNAKE`   | `DEFAULT_LAG = 2`                 |

### Stata 文件

| 对象   | 规范                                | 示例              |
| ------ | ----------------------------------- | ----------------- |
| 文件名 | `NN_model_name.do`（两位数字开头）  | `04_did_basic.do` |
| 全局宏 | `UPPER` 或 `lower` 均可，但全文一致 | `global root`     |
| 局部宏 | `lower_case`                        | `local varlist`   |

### Notebook 文件

| 对象   | 规范                                                                        | 示例                                     |
| ------ | --------------------------------------------------------------------------- | ---------------------------------------- |
| 文件名 | `类型前缀NN_模型_中文描述.ipynb`                                            | `T04_DiD_最低工资政策对就业的影响.ipynb` |
| 前缀   | `T`=传统计量，`ML`=机器学习，`DL`=深度学习，`RL`=强化学习，`LLM`=大模型应用 | `ML03_DoubleML_xxx.ipynb`                |

---

## 六、禁止生成的内容

以下内容 **Copilot 不得生成**，即使用户要求：

1. ❌ 在任何 Python 文件中使用 `tensorflow`、`keras`（统一用 PyTorch）
2. ❌ 在任何 Python 文件中使用旧版 `gym`（统一用 `gymnasium`）
3. ❌ 在 `empirlab/` 下的任何文件里写 `matplotlib.pyplot.show()`（图表由 utils/visualization.py 统一管理）
4. ❌ 在模型文件里硬编码数据路径（如 `pd.read_csv("C:/data/xxx.csv")`）
5. ❌ 跨模型文件互相 import（如 `from empirlab.traditional.ols import OLS` 出现在 `empirlab/traditional/iv.py` 中）
6. ❌ 在 notebooks/ 里重写已有的模型类
7. ❌ 生成任何 `requirements.txt` 以外版本的依赖（不允许随意升降级）
8. ❌ 创建不在项目结构规则内的目录（如 `src/`、`lib/`、`code/`）
9. ❌ 修改 `.github/copilot-instructions.md` 本文件（除非项目负责人明确要求）

---

## 七、允许 Copilot 自主完成的任务

以下情况 Copilot 可以在不确认的情况下生成：

✅ 在已有模型类中补全 `fit()` / `predict()` / `summary()` 方法的实现  
✅ 在 `utils/` 下添加新的工具函数  
✅ 在 `if __name__ == "__main__":` 块中补全示例代码  
✅ 为已有类和方法生成 docstring  
✅ 在 `stata/` 对应文件中补全注释和稳健性检验步骤  
✅ 在 notebook 中补全某个已确认步骤的代码（不能添加新的步骤结构）  
✅ 在 `tests/` 下生成单元测试

---

## 八、Go 工具链规则

- Go 代码只做**数据工程**：下载、清洗、格式转换、CLI
- Go 代码**不做统计建模**，不引入任何统计库
- 每个 CLI 工具在 `gotools/cmd/` 下有独立的 `main.go`
- 公共逻辑放在 `gotools/pkg/` 对应包中
- 所有 CLI 工具必须支持 `--help` 参数

---

## 九、快速检查清单（提交前自检）

新增或修改文件时，对照以下清单：

- [ ] 模型文件放在正确的子包目录下
- [ ] 类名与文件名对应
- [ ] 有完整的 docstring（参数、返回值、示例）
- [ ] `if __name__ == "__main__":` 存在且可运行
- [ ] 没有硬编码路径
- [ ] 没有跨模型文件的 import
- [ ] 如有对应 Stata 版本，`.do` 文件已创建
- [ ] 如有对应 Notebook，六段式结构完整，已 Run All
- [ ] 新依赖已加入 `requirements.txt`

---

_此文件由项目负责人维护，Copilot 不得修改。最后更新：2025-05_
