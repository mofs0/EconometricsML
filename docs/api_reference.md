# API 参考文档

> 本文档覆盖 `empirlab` 所有子包的公开接口。

---

## empirlab.traditional

### OLS

```python
class empirlab.traditional.ols.OLS(fit_intercept=True, robust=False)
```

普通最小二乘回归，支持经典标准误和 HC1 稳健标准误。

**参数**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `fit_intercept` | bool | True | 是否自动加入截距 |
| `robust` | bool | False | True 时使用 HC1 稳健标准误 |

**主要方法**

| 方法 | 说明 |
|------|------|
| `fit(X, y)` | 拟合模型，返回 self |
| `predict(X)` | 线性预测 |
| `residuals()` | 返回训练残差 |
| `summary()` | 返回完整推断字典 |
| `summary_table()` | 返回系数推断 DataFrame |

---

### IV2SLS

```python
class empirlab.traditional.iv.IV2SLS(robust=False)
```

工具变量两阶段最小二乘。

**主要方法**：`fit(endog_df, X, D, Z)`, `summary()`, `summary_table()`

---

### PanelFE

```python
class empirlab.traditional.panel.PanelFE(effects='two-way')
```

面板固定效应，支持个体效应、时间效应、双向效应。

---

### DiD

```python
class empirlab.traditional.did.DiD()
```

双重差分法，支持平行趋势检验。

---

### RDD

```python
class empirlab.traditional.rd.RDD(bandwidth=None, kernel='triangular')
```

断点回归设计，支持局部线性估计和 IK 最优带宽。

---

### PSM

```python
class empirlab.traditional.psm.PSM(method='nearest', caliper=0.05)
```

倾向得分匹配，支持近邻匹配和卡钳匹配。

---

### EventStudy

```python
class empirlab.traditional.event_study.EventStudy(window=(-10, 10), model='market')
```

事件研究法，估计超额收益（AR）和累计超额收益（CAR）。

---

### GARCH11

```python
class empirlab.traditional.garch.GARCH11()
```

GARCH(1,1) 波动率模型，MLE 估计。

---

### VAR

```python
class empirlab.traditional.var.VAR(lags=2)
```

向量自回归，支持脉冲响应函数（IRF）和预测方差分解（FEVD）。

---

### SyntheticControl

```python
class empirlab.traditional.synthetic_control.SyntheticControl()
```

合成控制法，二次规划求解最优权重。

---

## empirlab.ml

### DoubleML

```python
class empirlab.ml.double_ml.DoubleML(ml_y=None, ml_d=None, n_folds=5, random_state=42)
```

Double/Debiased Machine Learning 处理效应估计。

**参数**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `ml_y` | sklearn 回归器 | GradientBoosting | 拟合 E[Y\|X] |
| `ml_d` | sklearn 回归器 | RidgeCV | 拟合 E[D\|X] |
| `n_folds` | int | 5 | Cross-fitting 折数 |

**主要方法**：`fit(X, D, Y)`, `summary()`

**主要属性**：`theta_`, `se_`, `t_stat_`, `p_value_`, `ci_`

---

### LassoSelect

```python
class empirlab.ml.lasso_select.LassoSelect(cv=5, standardize=True, post_ols=True)
```

LASSO 变量筛选 + Post-LASSO OLS。

**主要方法**：`fit(X, y)`, `summary()`

**主要属性**：`selected_vars_`, `alpha_`, `lasso_coef_`, `post_coef_`

---

### RFRegressor

```python
class empirlab.ml.random_forest.RFRegressor(n_estimators=300, ...)
```

随机森林回归，含 OOB 评分和变量重要性。

---

### CausalForest

```python
class empirlab.ml.causal_forest.CausalForest(learner='T', base_model=None)
```

Meta-Learner 因果森林，估计 CATE 和 ATE。

**参数**：`learner`（"T" 或 "S"）

**主要方法**：`fit(X, D, Y, n_bootstrap=200)`, `summary()`, `cate_summary()`

---

## empirlab.dl

### LSTMForecaster

```python
class empirlab.dl.lstm_forecast.LSTMForecaster(lookback=20, horizon=1, hidden=64, ...)
```

LSTM 时间序列预测器（PyTorch 实现）。

**主要方法**：`fit(series, verbose=False)`, `predict(context)`

---

### TCNForecaster

```python
class empirlab.dl.tcn_forecast.TCNForecaster(lookback=30, n_channels=[32,32,32], ...)
```

时序卷积网络预测器（膨胀因果卷积）。

---

### MLPRegressor

```python
class empirlab.dl.mlp_regression.MLPRegressor(hidden_dims=[128,64,32], ...)
```

多层感知机回归（含 EarlyStopping）。

---

## empirlab.rl

### PortfolioEnv

```python
class empirlab.rl.portfolio_env.PortfolioEnv(returns, window=20, ...)
```

投资组合交易环境（Gymnasium 兼容接口）。

---

### DQNAgent

```python
class empirlab.rl.dqn_agent.DQNAgent(obs_dim, n_actions, ...)
```

Deep Q-Network 交易代理。

**主要方法**：`select_action(obs)`, `update()`, `train(env, n_episodes)`

---

## empirlab.llm

### TextToRegression

```python
class empirlab.llm.text_to_regression.TextToRegression(mode='rule')
```

自然语言 → 回归代码生成器。

**主要方法**：`generate(description, y_var, x_vars, ...)`

---

### SentimentIndex

```python
class empirlab.llm.sentiment_index.SentimentIndex(method='lm', lang='auto')
```

文本情感指数构建（LM 词典 / SnowNLP / BERT）。

**主要方法**：`score(texts)`, `score_df(df, text_col)`
