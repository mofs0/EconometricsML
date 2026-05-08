# empirlab — Python 实证方法库

`empirlab` 是本仓库的核心 Python 包，提供经济学、金融学、商科实证研究中常用方法的工程化实现。

## 安装

```bash
pip install -e .   # 从仓库根目录安装（开发模式）
```

## 子包结构

### traditional/ — 传统计量方法

| 文件 | 方法 | 对应 Notebook |
|------|------|--------------|
| `ols.py` | OLS 完整推断（SE/t/p/F/HC1/adj-R²） | T01 |
| `iv.py` | 工具变量 / 2SLS | T03 |
| `logit.py` | Logit / Probit 离散选择模型 | T02 |
| `panel.py` | 面板固定效应 / 随机效应 | T04 |
| `did.py` | 双重差分（DID） | T05 |
| `rd.py` | 回归断点设计（RD） | T06 |
| `psm.py` | 倾向得分匹配（PSM） | T07 |
| `event_study.py` | 事件研究 / 累计超额收益（CAR） | T08 |
| `garch.py` | GARCH 波动率建模 | T09 |
| `var.py` | VAR / 格兰杰因果 / 脉冲响应函数 | T10 |
| `synthetic_control.py` | 合成控制法 | T11 |

### ml/ — 机器学习因果推断

| 文件 | 方法 |
|------|------|
| `double_ml.py` | DoubleML（Neyman 正交，交叉拟合） |
| `causal_forest.py` | 因果森林（CATE 估计） |
| `regularized.py` | Lasso / Ridge / ElasticNet |
| `tree_models.py` | 随机森林 / GBDT |
| `high_dim_iv.py` | 高维工具变量 |
| `synthetic_did.py` | 合成 DID |
| `factor_model.py` | 因子模型 |
| `conformal_pred.py` | 共形预测区间 |
| `text_ml.py` | 文本特征 + ML |

### dl/ — 深度学习（PyTorch）

| 文件 | 方法 |
|------|------|
| `mlp_regressor.py` | 多层感知机回归 |
| `mlp.py` | 通用 MLP 骨架 |

### rl/ — 强化学习（Gymnasium）

| 文件 | 方法 |
|------|------|
| `q_learning_agent.py` | Q-Learning 智能体 |

### llm/ — LLM 调用封装

| 文件 | 功能 |
|------|------|
| `prompt_client.py` | 大模型 API 调用封装 |

### utils/ — 工具函数

统一管理：指标计算（metrics）、可视化（visualization）、数据预处理（preprocessing）、数据读写（data_io）。

## 设计原则

- 每个文件：独立类 + 完整 docstring + `if __name__ == "__main__"` 最小可运行示例
- 模型类只放 `empirlab/` 对应子包，Notebook 只调用，不重写
- 深度学习统一 PyTorch，强化学习统一 Gymnasium
