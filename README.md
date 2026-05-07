# EmpirLab

**面向定量研究的工程化代码仓库。**

当前阶段聚焦**经济学 · 金融学 · 商科**实证研究；
架构预留其他学科（医学统计、社会学等）的扩展入口，后续按需开发。

> Python 包名：`empirlab` · Git 仓库：`EconometricsML`（计划改名为 `empirlab`）

---

## 1. 仓库定位

提供可复用、可审计、可扩展的实证研究模板：

- 传统计量方法（OLS / IV / Panel / DiD / RD / GARCH …）
- 机器学习因果推断（DoubleML / CausalForest / Regularized …）
- 深度学习时序与预测（PyTorch）
- 强化学习（Gymnasium 接口）
- LLM 应用封装（调用，非训练）
- Stata 对照实现

---

## 2. 项目结构

```text
empirlab/
├── traditional/          # 经典计量统计方法
│   ├── ols.py            ✅ 完整推断（SE/t/p/F/HC1/adj-R²）
│   ├── iv.py
│   ├── panel.py
│   ├── did.py
│   ├── rd.py
│   ├── psm.py
│   ├── synthetic_control.py
│   ├── event_study.py
│   ├── garch.py
│   └── var.py
├── ml/                   # 机器学习方法
│   ├── double_ml.py      ✅
│   ├── causal_forest.py
│   ├── regularized.py
│   ├── tree_models.py
│   └── ...（共 10 个）
├── dl/                   # 深度学习（统一 PyTorch）
│   ├── mlp_regressor.py  ✅
│   └── mlp.py
├── rl/                   # 强化学习（统一 Gymnasium）
│   └── q_learning_agent.py ✅
├── llm/                  # LLM 调用封装
│   ├── prompt_client.py  ✅
│   └── ...（共 5 个）
└── utils/                # 工具函数（metrics / visualization / preprocessing / data_io）✅

notebooks/
├── T01_OLS_Mincer工资方程.ipynb     ✅ 完整六段式论文示例
├── traditional/
├── ml/
├── dl/
├── rl/
└── llm/

stata/
├── traditional/
│   ├── 01_ols.do         ✅ 完整 Mincer 示例
│   └── 02_iv_2sls.do
└── ml/
    └── 01_double_ml.do   ✅

gotools/
└── cmd/csvpeek/          ✅ CSV 预览 CLI
```

---

## 3. 快速开始

```bash
pip install -r requirements.txt
pip install -e .

# 直接运行最小示例（无需额外数据）
python empirlab/traditional/ols.py
python empirlab/ml/double_ml.py
python empirlab/dl/mlp_regressor.py
python empirlab/rl/q_learning_agent.py
python empirlab/llm/prompt_client.py
```

---

## 4. 使用示例

### OLS（完整推断）

```python
from empirlab.traditional.ols import OLS, mincer_data

df = mincer_data(n=500)
model = OLS(robust=True).fit(df[["educ", "exper", "exper2"]], df["ln_wage"])
print(model.summary_table().round(4))
# coef / se(HC1) / t / p_value / ci_lower / ci_upper
```

### DoubleML

```python
import numpy as np
from empirlab.ml import DoubleML

rng = np.random.default_rng(42)
n, X = 400, rng.standard_normal((400, 5))
d = 0.4 * X[:, 0] + rng.standard_normal(n) * 0.3
y = 1.2 * d + 0.5 * X[:, 0] + rng.standard_normal(n) * 0.5
model = DoubleML().fit(X, y, d)
print(model.summary())
```

---

## 5. Notebook 规范（六段式）

每个 `.ipynb` 必须包含以下六段标题，顺序不变：

```
## 0. 论文信息
## 1. 研究设计与识别策略
## 2. 数学理论与模型
## 3. 数据加载与预处理
## 4. 模型估计
## 5. 结果解读与稳健性检验
## 6. 可视化
```

已完成示例：[T01_OLS_Mincer工资方程.ipynb](notebooks/T01_OLS_Mincer工资方程.ipynb)

---

## 6. 近期开发计划（经济 · 金融 · 商科）

### 方法层（empirlab/）

| 优先级 | 文件 | 说明 |
|-------|------|------|
| 🔴 高 | `traditional/logit.py` | Logit / Probit，离散选择模型 |
| 🔴 高 | `traditional/var.py` 补全 | VAR，宏观预测标配 |
| 🟡 中 | `dl/lstm.py` | 时序预测，金融应用广 |
| 🟡 中 | `dl/transformer_ts.py` | 时序 Transformer |
| 🟢 低 | `rl/dqn.py` | 强化学习交易策略 |
| 🟢 低 | `rl/ppo.py` | 投资组合优化 |

### Notebook 层（按论文主题）

| 优先级 | 文件 | 对应方法 |
|-------|------|---------|
| 🔴 高 | `T02_IV_教育回报工具变量.ipynb` | IV / 2SLS |
| 🔴 高 | `T03_DiD_政策评估.ipynb` | Difference-in-Differences |
| 🔴 高 | `T04_Panel_企业固定效应.ipynb` | Panel FE / RE |
| 🟡 中 | `ML01_DoubleML_因果效应估计.ipynb` | DoubleML |
| 🟡 中 | `DL01_LSTM_股价预测.ipynb` | LSTM |
| 🟢 低 | `RL01_DQN_交易策略.ipynb` | DQN |

---

## 7. 其他学科（预留，未开发）

架构已预留以下学科的扩展入口，后续按需开发：

- **生物统计 / 医学**：生存分析（Cox PH、KM）、随机对照试验、荟萃分析
- **社会学 / 政策**：调查方法、多层次模型
- **管理学**：结构方程模型（SEM）

如需扩展，新增 `empirlab/bio/`、`notebooks/bio/`、`stata/bio/` 即可，
无需改动现有代码。

---

## 8. 代码规范

详见 [`.github/copilot-instructions.md`](.github/copilot-instructions.md)。核心约束：

- 模型类只放 `empirlab/` 对应子包，不在 notebook 里重写
- 每个文件：独立类 + 完整 docstring + `if __name__ == "__main__"` 最小示例
- 深度学习统一 PyTorch，强化学习统一 Gymnasium
- `utils/` 统一管理工具函数

---

## 9. Stata 对应

| Python | Stata | 状态 |
|--------|-------|------|
| `traditional/ols.py` | `stata/traditional/01_ols.do` | ✅ 完整 |
| `traditional/iv.py` | `stata/traditional/02_iv_2sls.do` | 骨架 |
| `ml/double_ml.py` | `stata/ml/01_double_ml.do` | ✅ 完整 |

---

## 10. Go 工具链

```bash
cd gotools && go run ./cmd/csvpeek --help
```

---

## 11. 许可证

MIT License
