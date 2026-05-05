# 计量经济学 Notebook 模板指南

## 参考文件

**Exemplar**: `exemplar_OLS_Angrist2008.ipynb`  
这是所有计量经济学方法 notebook 的模板参考。

---

## Notebook 结构（必须遵循）

每个 notebook 应包含以下七个部分：

### 1. **标题与论文引用** (Markdown)
```markdown
# 方法名称详细教程
## 基于 [作者] ([年份]) 的方法论

**参考论文：**
- [完整引文，确保真实可查]
  - 关键章节或页码
  - 核心贡献说明
```

### 2. **第一部分：理论基础** (Markdown)
- 线性模型的设置（数学公式）
- 该方法的定义和含义
- 作者的关键见解（为什么这个方法重要）
- 大样本性质和渐近分布

### 3. **第二部分：数据生成与准备** (Python + Markdown)
- 生成 synthetic data（DGP 要反映论文设定）
- 显示数据的前几行和描述统计
- 说明数据为什么能演示这个方法

### 4. **第三部分：模型拟合** (Python)
- 从 `econml` 导入相关类
- 初始化和拟合模型
- 显示系数估计值与真实值的对比

### 5. **第四部分：模型诊断** (Python + 可视化)
- 计算 R²、残差统计
- 绘制诊断图（残差 vs 拟合值、Q-Q 图等）
- 根据论文标准解释诊断结果

### 6. **第五部分：统计推断** (Python)
- 计算标准误、t 统计量、p 值
- 构建置信区间
- 创建结果表（Regression Table 风格）

### 7. **第六部分：因果解释与局限** (Markdown + 演示)
- 该方法的假设条件
- 什么时候可以因果解释
- 常见陷阱（OVB, multicollinearity 等）
- 如有适用，演示假设违反的影响

### 8. **第七部分：如何修改库代码** (Markdown + 代码示例)
- 展示库代码的核心实现
- 列举常见的修改场景
- 提供修改示例

---

## 注释与双语文本风格

### ❌ 不要：混合风格（中英混杂）
```python
# 计算 beta 值（calculation of coefficients）
beta = (X.T @ X) ** (-1) @ X.T @ y  # OLS solution
```

### ✅ 要：分离风格

```python
"""
计算普通最小二乘法的系数估计值。
该部分实现了 OLS 的闭形式解：β_hat = (X'X)^{-1} X'y
这是线性回归的核心计算步骤。
"""

"""
Compute the Ordinary Least Squares coefficient estimates.
This section implements the closed-form OLS solution: β_hat = (X'X)^{-1} X'y
This is the core computational step of linear regression.
"""

beta = (X.T @ X) ** (-1) @ X.T @ y
```

---

## 论文列表（已验证真实性）

### 计量经济学模块

| # | 方法 | 论文引用 | 出版社/期刊 | 说明 |
|---|------|--------|----------|------|
| 1 | OLS | Angrist, J. D., & Pischke, J. S. (2008). *Mostly Harmless Econometrics*. Princeton University Press. Ch. 2 | 经典教材 | 因果推断基础 |
| 2 | IV | Angrist, J. D., & Pischke, J. S. (2009). "The Credibility Revolution in Empirical Economics." *Journal of Economic Literature*, 48(1), 3-28. | JEL | 工具变量详论 |
| 3 | DiD | Card, D., & Krueger, A. B. (1994). "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania." *American Economic Review*, 84(4), 772-793. | AER | 经典双重差分应用 |
| 4 | Fixed Effects | Hsiao, C. (2003). *Analysis of Panel Data* (2nd ed.). Cambridge University Press. Ch. 3 | 教材 | 面板数据固定效应 |
| 5 | Random Effects | Hsiao, C. (2003). *Analysis of Panel Data* (2nd ed.). Cambridge University Press. Ch. 4 | 教材 | 面板数据随机效应 |
| 6 | Logit/Probit | McFadden, D. (1974). "Conditional logit analysis of qualitative choice behavior." In *Frontiers of Econometrics* (Ed. Zarembka, P.). Academic Press. | 经典 | 二值选择模型 |
| 7 | GMM | Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica*, 50(4), 1029-1054. | Econometrica | GMM 理论基础 |
| 8 | ARIMA | Box, G. E., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day. | 经典教材 | 时间序列 ARIMA |
| 9 | VAR | Sims, C. A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1-48. | Econometrica | VAR 模型 |
| 10 | GARCH | Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327. | JoE | GARCH 模型基础 |

### 机器学习模块

| # | 方法 | 论文引用 | 出版社/期刊 | 说明 |
|---|------|--------|----------|------|
| 1 | Ridge | Hoerl, A. E., & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, 12(1), 55-67. | Technometrics | Ridge 回归 |
| 2 | Lasso | Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society*, 58(1), 267-288. | JRSS-B | Lasso 基础 |
| 3 | Random Forest | Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32. | ML Journal | 随机森林 |
| 4 | Gradient Boosting | Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189-1232. | Annals Stat | 梯度提升 |
| 5 | SVM | Cortes, C., & Vapnik, V. (1995). "Support-Vector Networks." *Machine Learning*, 20(3), 273-297. | ML Journal | SVM 基础 |
| 6 | KNN | Fix, E., & Hodges, J. L. (1951). "Discriminatory Analysis. Nonparametric Discrimination: Consistency Properties." USAF School of Aviation Medicine, Randolph Field, Texas. | 技术报告 | KNN 最早提出 |
| 7 | K-means | Lloyd, S. (1957). "Least Squares Quantization in PCM." Bell Labs Technical Journal, 28(2), 129-137. | Bell Labs | K-means 聚类 |
| 8 | PCA | Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer. | 教材 | PCA 综合 |
| 9 | XGBoost | Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. | KDD | XGBoost 论文 |
| 10 | Logistic Regression | Cox, D. R. (1958). "The Regression Analysis of Binary Sequences." *Journal of the Royal Statistical Society*, 20(2), 215-242. | JRSS-B | Logistic 回归基础 |

### 深度学习模块

| # | 方法 | 论文引用 | 出版社/期刊 | 说明 |
|---|------|--------|----------|------|
| 1 | MLP | Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization." *Psychological Review*, 65(6), 386-408. | 经典 | 感知器原理 |
| 2 | CNN | LeCun, Y., Bouvou, B., Bengio, Y., & Haffner, P. (1998). "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*, 86(11), 2278-2324. | IEEE | CNN 经典 |
| 3 | RNN | Hopfield, J. J. (1982). "Neural Networks and Physical Systems with Emergent Collective Computational Abilities." *Proceedings of the National Academy of Sciences*, 79(8), 2554-2558. | PNAS | RNN 基础 |
| 4 | LSTM | Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780. | Neural Comp | LSTM 论文 |
| 5 | Transformer | Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems (NIPS)*, 30. | NeurIPS | Transformer 论文 |
| 6 | Autoencoder | Hinton, G. E., & Zemel, R. S. (1994). "Autoencoders, Minimum Description Length and Helmholtz Free Energy." *Advances in Neural Information Processing Systems*, 6. | NeurIPS | Autoencoder 基础 |

### 强化学习模块

| # | 方法 | 论文引用 | 出版社/期刊 | 说明 |
|---|------|--------|----------|------|
| 1 | Q-Learning | Watkins, C. J., & Dayan, P. (1992). "Q-Learning." *Machine Learning*, 8(3-4), 279-292. | ML Journal | Q-Learning 基础 |
| 2 | SARSA | Rummery, G. A., & Niranjan, M. (1994). "On-Line Q-Learning Using Connectionist Systems." *Technical Report CUED/F-INFENG/TR 166*. Cambridge University Engineering Dept. | 技术报告 | SARSA 论文 |
| 3 | DQN | Mnih, V., et al. (2015). "Human-level Control Through Deep Reinforcement Learning." *Nature*, 529(7587), 529-533. | Nature | DQN 突破 |
| 4 | PPO | Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*. | arXiv | PPO 论文 |
| 5 | A2C/A3C | Mnih, V., et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." *Proceedings of the 33rd International Conference on Machine Learning*, 48. | ICML | A3C 论文 |

---

## 创建新 Notebook 的步骤

1. **复制 Exemplar 的结构**：参考 `exemplar_OLS_Angrist2008.ipynb`
2. **选定论文**：从上面的表格中选择，确保论文真实可查
3. **生成合适的 DGP**：synthetic data 应该反映论文的设定
4. **编写中文讲解**：论文方法论的完整中文讲述
5. **编写英文讲解**：完整的英文翻译（分离，不混合）
6. **库代码展示**：如何使用 `econml` 中的实现
7. **完整工作流**：从数据到诊断到解释的全过程
8. **提交前检查**：
   - [ ] 论文引用无误、真实存在
   - [ ] 所有代码都能运行
   - [ ] 中英文分离，无混合 `(中英)` 标记
   - [ ] 至少包含一张诊断图
   - [ ] 包含统计推断结果

---

## 论文来源说明

所有论文都来自：
- **顶级期刊**：Econometrica, American Economic Review, Journal of Econometrics, Machine Learning, Neural Computation
- **经典教科书**：如 Angrist & Pischke, Hsiao, Box & Jenkins
- **著名国际学术会议**：NeurIPS, ICML, KDD

学生和研究人员可以通过：
- Google Scholar (scholar.google.com)
- 机构数据库 (如果有学术机构访问权限)
- ResearchGate/Academia.edu (通常作者会上传预印本)

确保所有引用都是 **可验证的** 和 **学术规范的**。
