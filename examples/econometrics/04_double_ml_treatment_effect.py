"""
示例 04：Double ML —— 高维控制下的处理效应估计
================================================
场景：企业享受政府补贴（D）对全要素生产率（Y）的影响
混淆变量（X）：企业规模、杠杆、年龄、行业等高维特征

对比：
  1. 朴素 OLS（可能有遗漏变量偏误）
  2. Double ML（交叉拟合 + Neyman 正交，渐近无偏）

运行：
    python examples/econometrics/04_double_ml_treatment_effect.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empirlab.ml.double_ml import DoubleML, dml_data
from empirlab.traditional.ols import OLS

# ── 数据 ──────────────────────────────────────────────────────────────
TRUE_THETA = 0.3
df = dml_data(n=1200, p=10, true_theta=TRUE_THETA, seed=42)
X_cols = [c for c in df.columns if c.startswith("x")]
X = df[X_cols]
D = df["d"]
Y = df["y"]

print(f"样本量：{len(df)}, 控制变量维度：{len(X_cols)}, 真实处理效应：{TRUE_THETA}")

# ── 朴素 OLS ──────────────────────────────────────────────────────────
ols_all = OLS(robust=True).fit(pd.concat([X, D.rename("d")], axis=1), Y)
ols_d_coef = ols_all.summary()["coefficients"]["d"]["coef"]
print(f"\n朴素 OLS θ = {ols_d_coef:.4f}（偏误 = {ols_d_coef - TRUE_THETA:+.4f}）")

# ── Double ML ─────────────────────────────────────────────────────────
dml = DoubleML(n_folds=5)
dml.fit(X, D, Y)
print(f"Double ML θ = {dml.theta_:.4f}（偏误 = {dml.theta_ - TRUE_THETA:+.4f}）")
print(f"SE = {dml.se_:.4f}, t = {dml.t_stat_:.2f}, p = {dml.p_value_:.4f}")
print(f"95% CI：[{dml.ci_[0]:.4f}, {dml.ci_[1]:.4f}]")
print(dml.summary().to_string(index=False))

# ── 可视化：残差散点 ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(dml._resid_d, dml._resid_y, alpha=0.2, s=8, color="#4C72B0")
slope = dml.theta_
x_range = np.linspace(dml._resid_d.min(), dml._resid_d.max(), 100)
ax.plot(x_range, slope * x_range, color="#C44E52", linewidth=2,
        label=f"θ = {slope:.3f}（真值 = {TRUE_THETA}）")
ax.axhline(0, color="k", linewidth=0.5)
ax.axvline(0, color="k", linewidth=0.5)
ax.set_xlabel("D̃（处理残差）")
ax.set_ylabel("Ỹ（结果残差）")
ax.set_title("Double ML：残差-残差图")
ax.legend()
plt.tight_layout()
plt.savefig("examples/econometrics/04_dml_residuals.png", dpi=150)
print("图表已保存至 examples/econometrics/04_dml_residuals.png")
