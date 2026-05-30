"""
示例 05：LASSO 变量筛选 —— 高维回归中的稀疏变量选择
=====================================================
场景：从 50 个候选财务指标中自动筛选与企业绩效相关的真正重要变量
方法：LassoCV（自动选 λ）+ Post-LASSO OLS（去偏估计）

运行：
    python examples/econometrics/05_lasso_variable_selection.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empirlab.ml.lasso_select import LassoSelect, lasso_data

# ── 数据：50 个指标，真实只有 8 个起作用 ────────────────────────────
np.random.seed(0)
df = lasso_data(n=500, p=50, k_true=8, noise=1.5, seed=0)
y = df["y"]
X = df.drop(columns="y")
TRUE_VARS = {f"x{i+1}" for i in range(8)}
print(f"样本量：{len(df)}, 候选变量：{X.shape[1]}, 真实变量：{len(TRUE_VARS)}")

# ── LASSO 筛选 ────────────────────────────────────────────────────────
model = LassoSelect(cv=5, standardize=True, post_ols=True)
model.fit(X, y)

print(f"\n最优 λ = {model.alpha_:.6f}")
print(f"选中变量数：{len(model.selected_vars_)}")
print(f"选中变量：{sorted(model.selected_vars_)}")

# ── 准确率评估 ────────────────────────────────────────────────────────
selected = set(model.selected_vars_)
tp = len(TRUE_VARS & selected)
fp = len(selected - TRUE_VARS)
fn = len(TRUE_VARS - selected)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\n精确率（Precision）：{precision:.2%}")
print(f"召回率（Recall）：   {recall:.2%}")

# ── Post-LASSO OLS 系数 ───────────────────────────────────────────────
if model.post_coef_ is not None:
    print("\nPost-LASSO OLS 系数（活跃集）：")
    print(model.post_coef_.round(4).to_string())

# ── 可视化：系数路径 ──────────────────────────────────────────────────
summary_df = model.summary().head(15)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#C44E52" if v in TRUE_VARS else "#4C72B0" for v in summary_df.index]
bars = ax.barh(summary_df.index, summary_df["lasso_coef"], color=colors)
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlabel("LASSO 系数")
ax.set_title("LASSO 系数（红色 = 真实变量，蓝色 = 噪声变量）")

from matplotlib.patches import Patch
legend = [Patch(color="#C44E52", label="真实变量"),
          Patch(color="#4C72B0", label="噪声变量")]
ax.legend(handles=legend)
plt.tight_layout()
plt.savefig("examples/econometrics/05_lasso_coefs.png", dpi=150)
print("图表已保存至 examples/econometrics/05_lasso_coefs.png")
