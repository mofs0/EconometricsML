# EmpirLab

**面向定量研究的工程化代码仓库。**

聚焦**经济学 · 金融学 · 商科**实证研究，提供可复用的计量方法代码模板；
`academic` 分支另含论文写作全流程指南与数据获取服务。

> Python 包名：`empirlab` · Git 仓库：`EconometricsML`

---

## 1. 仓库定位

- **`main` 分支**：可复用、可审计、可扩展的实证研究代码模板（传统计量 + ML + DL + RL + LLM）
- **`academic` 分支**：论文写作从选题到投稿的完整指南（A01–A07），含中英文期刊写作、学术裁缝方法、数据获取服务

---

## 2. 项目结构

```text
EconometricsML/
├── empirlab/                    # Python 包（核心方法库）
│   ├── traditional/             # 经典计量方法
│   │   ├── ols.py               ✅ 完整推断（SE/t/p/F/HC1/adj-R²）
│   │   ├── iv.py                ✅ 2SLS / 工具变量
│   │   ├── panel.py             ✅ 固定效应 / 随机效应
│   │   ├── did.py               ✅ 双重差分
│   │   ├── rd.py                ✅ 回归断点
│   │   ├── psm.py               ✅ 倾向得分匹配
│   │   ├── event_study.py       ✅ 事件研究 / CAR
│   │   ├── garch.py             ✅ GARCH 波动率建模
│   │   ├── var.py               ✅ VAR / 脉冲响应函数
│   │   └── synthetic_control.py ✅ 合成控制法
│   ├── ml/                      # 机器学习方法
│   │   ├── double_ml.py         ✅ DoubleML（Neyman 正交）
│   │   ├── causal_forest.py     ✅ 因果森林
│   │   ├── regularized.py       ✅ Lasso / Ridge / ElasticNet
│   │   └── tree_models.py       ✅
│   ├── dl/                      # 深度学习（PyTorch）
│   │   ├── mlp_regressor.py     ✅
│   │   └── mlp.py               ✅
│   ├── rl/                      # 强化学习（Gymnasium）
│   │   └── q_learning_agent.py  ✅
│   ├── llm/                     # LLM 调用封装
│   │   └── prompt_client.py     ✅
│   └── utils/                   ✅ metrics / visualization / preprocessing / data_io
│
├── notebooks/
│   └── traditional/             # T01–T11 论文级示例 Notebook（全部完成）
│       ├── T01_OLS_Mincer工资方程.ipynb        ✅
│       ├── T01_OLS_最小可运行示例.ipynb         ✅
│       ├── T02_Logit_企业出口决策.ipynb         ✅
│       ├── T03_IV_制度质量与经济发展.ipynb      ✅
│       ├── T04_Panel_企业生产率.ipynb           ✅
│       ├── T05_DiD_最低工资政策.ipynb           ✅
│       ├── T06_RD_法定饮酒年龄与死亡率.ipynb    ✅
│       ├── T07_PSM_政府补贴研发效应.ipynb       ✅
│       ├── T08_EventStudy_股权质押公告效应.ipynb ✅
│       ├── T09_GARCH_A股波动率.ipynb            ✅
│       ├── T10_VAR_货币政策传导.ipynb           ✅
│       └── T11_SyntheticControl_政策效应.ipynb  ✅
│
├── academic/                    # 论文写作系列（academic 分支）
│   ├── A01_文献检索与高效阅读.ipynb
│   ├── A02_选题与开题报告.ipynb
│   ├── A03_期刊选择与投稿.ipynb    # 含 SCI/SSCI/CSSCI/北大核心/普刊
│   ├── A04_SCI小论文写作全指南.ipynb
│   ├── A05_毕业大论文.ipynb
│   ├── A06_学术裁缝写作方法.ipynb  # 情境迁移/机制叠加/测度改进/异质性扩展
│   └── A07_数据获取服务.ipynb      # 联系：1795837192@qq.com
│
├── stata/
│   ├── traditional/
│   │   ├── 01_ols.do             ✅ 完整 Mincer 示例
│   │   └── 02_iv_2sls.do
│   └── ml/
│       └── 01_double_ml.do       ✅
│
└── rerun_all_notebooks.ps1       # Windows 一键重跑 + 推送脚本
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

---

## 6. academic 分支：论文写作指南

切换分支后可访问 `academic/` 文件夹，包含 7 个专题 Notebook：

| 编号 | 主题 | 核心内容 |
|------|------|----------|
| A01 | 文献检索与高效阅读 | 平台选择、关键词策略、WWH 阅读框架、创新点挖掘 |
| A02 | 选题与开题报告 | 选题风险管理、五段式开题结构、答辩 Q&A |
| A03 | 期刊选择与投稿 | SCI/SSCI/