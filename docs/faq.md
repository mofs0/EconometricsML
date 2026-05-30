# 常见问题（FAQ）

---

## 安装与环境

**Q：安装依赖时报错 `ERROR: Could not find a version that satisfies the requirement torch`**

A：torch 需要根据你的 CUDA 版本安装。访问 [pytorch.org](https://pytorch.org/get-started/locally/) 选择对应版本。CPU 版本：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

**Q：`pip install -r requirements.txt` 报 `torch` 版本冲突**

A：先单独安装 torch，再安装其他依赖：
```bash
pip install torch>=2.0.0
pip install -r requirements-base.txt
```

`requirements-base.txt` 是不含 torch 的纯计量依赖。

---

**Q：中文字符在图表中显示为方块（□□□）**

A：matplotlib 默认不支持中文。解决方法：
```python
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'  # Windows
# matplotlib.rcParams['font.family'] = 'PingFang SC'    # macOS
# matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'  # Linux
matplotlib.rcParams['axes.unicode_minus'] = False
```

---

## 方法使用

**Q：Double ML 的 `theta_` 和 OLS 的处理变量系数相差很大，哪个对？**

A：理论上 Double ML 的估计更可靠。OLS 的系数在高维混淆下存在正则化偏误；Double ML 通过 Neyman 正交化消除了这个偏误。如果差异过大（> 0.5），说明混淆变量对处理变量有较强的非线性影响。

---

**Q：LASSO 选出的变量每次运行结果不一样**

A：LASSO 在变量相关时有随机性。建议：
1. 固定随机种子：`LassoSelect(cv=5)` 内部已固定 `random_state=42`
2. 多次运行取交集（稳定变量）
3. 使用 Elastic Net 降低多重共线性的影响

---

**Q：因果森林的 T-Learner 和 S-Learner 哪个更好？**

A：
- **T-Learner** 通常更好，允许两组有完全不同的预测函数
- **S-Learner** 在处理率极不平衡时（如 < 5% 处理组）更稳定
- 建议同时跑两个，结果相近说明估计稳健

生产级实现建议使用 `econml` 的 `CausalForestDML`（需 `pip install econml`）。

---

**Q：LSTM 训练 loss 不下降 / 震荡**

A：常见原因和解决方案：
1. 学习率太大：改为 `lr=1e-4`
2. 没有标准化：`LSTMForecaster` 内部已做 Z-score，确认没有手动再标准化
3. 序列太短：建议 `lookback >= 20`
4. 梯度爆炸：代码已加 `clip_grad_norm_`，若仍然爆炸减小 `lr`

---

**Q：`empirlab.rl` 的 PortfolioEnv 为什么只有 3 个离散动作？**

A：这是教学简化版。实际应用中，动作空间通常是连续的（n_assets 维权重向量），需要配合 PPO / DDPG 等策略梯度算法。离散版本适合入门理解 RL 框架，连续版本见 [Jiang et al. 2017](https://arxiv.org/abs/1706.10059)。

---

## 学术写作

**Q：论文方法部分如何描述使用了 Double ML？**

A：参考写法（可直接引用）：

> 为控制高维混淆变量对处理效应估计的偏误，本文采用 Chernozhukov et al.（2018）提出的双重去偏机器学习方法（Double/Debiased ML，DML）。该方法通过 K 折交叉拟合将模型训练与推断样本分离，并利用 Neyman 正交性条件消除 nuisance 函数估计误差对目标参数的一阶影响，从而实现 √n 一致性的处理效应估计。

---

**Q：审稿人问"为什么用随机森林而不是 OLS 作为 nuisance 估计器"？**

A：DML 对 nuisance 估计器的选择是灵活的——任何满足收敛速度条件（通常要求 n^{-1/4}）的估计器均可。随机森林相比 OLS 的优势在于：①无需指定函数形式；②自动捕捉非线性和交互效应；③在高维情形下更鲁棒。具体选择可通过 AIC/BIC 或 CV R² 验证 nuisance 拟合质量。

---

## 数据获取

**Q：如何获取 A 股财务数据？**

A：参见 `academic/A07_数据获取服务.ipynb`，或联系：1795837192@qq.com

**Q：可以用 akshare 直接拉数据吗？**

A：可以。以下是常用接口：
```python
import akshare as ak

# 沪深300成分股财务数据
df = ak.stock_financial_abstract(symbol="000001")

# 宏观数据（GDP、CPI）
gdp = ak.macro_china_gdp_yearly()
cpi = ak.macro_china_cpi_monthly()
```
