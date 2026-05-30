# 机器学习计量方法使用指南

> 什么时候用 ML？怎么用对？避免哪些陷阱？

---

## 为什么计量经济学需要机器学习

传统计量方法（OLS/IV/DID）在低维、线性、理论驱动的场景下表现优秀。但面对以下问题时会遇到瓶颈：

| 问题 | 传统方法的局限 | ML 解决方案 |
|------|--------------|------------|
| 50+ 个候选控制变量 | 难以手动筛选，多重共线性 | LASSO 稀疏选择 |
| 非线性混淆关系 | 线性假设可能偏误 | Random Forest 非参数估计 |
| 高维 nuisance 函数 | OLS 正则化偏误 | Double ML |
| 异质处理效应 | 只能估计平均效应 | Causal Forest |
| 时间序列长程依赖 | ARIMA 感受野有限 | LSTM / TCN |

---

## 核心原则：预测 ≠ 因果

**这是最容易犯的错误。**

- `sklearn.LinearRegression` 预测 R² 高，不代表系数有因果解释
- 随机森林的变量重要性 ≠ 因果效应大小
- 神经网络预测准确，不代表特征对结果有影响

**什么时候 ML 可以用于因果推断？**

只有在以下设计中，ML 的估计结果才有因果含义：
1. **Double ML**：利用 Neyman 正交条件消除正则化偏误
2. **Causal Forest**：基于随机化或条件独立性假设
3. **ML 辅助工具变量**：用 ML 预测工具变量强度（Angrist & Frandsen 2022）

---

## 各方法适用场景

### Double ML（`empirlab.ml.DoubleML`）

**适用**：
- 处理效应估计（D → Y），同时存在高维混淆变量 X
- 例：政府补贴（D）对企业绩效（Y）的影响，控制50个财务指标（X）

**不适用**：
- D 不存在 / 无明确处理变量
- 样本量 < 200（交叉拟合失效）

```python
from empirlab.ml import DoubleML, dml_data

df = dml_data(n=1000, p=20, true_theta=0.5)
X = df[[c for c in df.columns if c.startswith("x")]]
model = DoubleML(n_folds=5).fit(X, df["d"], df["y"])
print(f"θ = {model.theta_:.4f}, p = {model.p_value_:.4f}")
```

**注意**：
- `n_folds=5` 是默认推荐；样本 < 500 可用 `n_folds=3`
- `ml_y` 和 `ml_d` 的模型选择影响有限效率，不影响一致性

---

### LASSO 变量筛选（`empirlab.ml.LassoSelect`）

**适用**：
- 从 p > n 或 p 接近 n 的高维变量集合中选出真正重要的变量
- 稳健性检验：用 LASSO 筛选的变量做 OLS 对比

**不适用**：
- 理论已有明确的控制变量集合（无需数据驱动选择）
- 高度共线性变量（LASSO 只选其中一个，解释有歧义）

```python
from empirlab.ml import LassoSelect, lasso_data

df = lasso_data(n=400, p=30, k_true=6)
model = LassoSelect(cv=5, post_ols=True).fit(df.drop("y",axis=1), df["y"])
print("选中变量：", model.selected_vars_)
print(model.summary())
```

**注意**：
- 总是开启 `post_ols=True`，LASSO 系数本身有收缩偏误
- `standardize=True` 是默认推荐（系数可比较）

---

### Causal Forest（`empirlab.ml.CausalForest`）

**适用**：
- 异质处理效应（CATE）：不同个体对政策的响应差异
- 精准政策设计：识别哪类人群受益最大

**不适用**：
- 只关心平均处理效应（ATE），DiD/IV 已足够
- 样本量 < 500（估计方差太大）

```python
from empirlab.ml import CausalForest, cf_data

df = cf_data(n=800, true_ate=1.0, heterogeneous=True)
model = CausalForest(learner="T").fit(df[["x1","x2","x3"]], df["d"], df["y"])
print(model.cate_summary())
```

---

## 模型选择建议

```
研究问题
  ├── 因果推断（处理效应）
  │     ├── 低维控制变量（< 20）→ DiD / IV / PSM
  │     ├── 高维控制变量（20+）→ Double ML
  │     └── 关心异质性       → Causal Forest
  │
  ├── 变量筛选 / 预测
  │     ├── 线性稀疏          → LASSO
  │     ├── 非线性            → Random Forest / XGBoost
  │     └── 时间序列          → LSTM / TCN
  │
  └── 文本数据
        ├── 情感指数          → SentimentIndex（LM 词典）
        └── 代码生成          → TextToRegression
```

---

## 常见陷阱与处理

**过拟合**：始终用 OOB 分数或 CV 评估，而非训练集 R²

**数据泄露**：预测未来时，确保特征只用过去信息

**多重比较**：用 ML 筛出的变量，在独立样本上验证

**解释性**：向审稿人解释 ML 方法时，强调 Neyman 正交性（DML）或半参数理论（CF）
