"""
示例 01：OLS Mincer 工资方程
=============================
复现 Mincer (1974) 教育回报率研究设计：
    ln_wage = β0 + β1*educ + β2*exper + β3*exper² + ε

运行：
    python examples/econometrics/01_ols_wage_equation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
from empirlab.traditional.ols import OLS, mincer_data

# ── 生成数据 ──────────────────────────────────────────────────────────
df = mincer_data(n=1000, seed=42)
print(f"样本量 N = {len(df)}")
print(df.describe().round(3))

# ── OLS 估计（HC1 稳健标准误）────────────────────────────────────────
model = OLS(robust=True).fit(df[["educ", "exper", "exper2"]], df["ln_wage"])
print("\n" + "="*60)
print("OLS 估计结果（HC1 稳健标准误）")
print("="*60)
print(model.summary_table().round(4))

s = model.summary()
print(f"\nR² = {s['fit']['R2']:.4f}, Adj-R² = {s['fit']['adj_R2']:.4f}")
print(f"F 统计量 = {s['fit']['F_stat']:.2f}, p = {s['fit']['F_pval']:.4e}")

# ── 教育回报率解释 ─────────────────────────────────────────────────────
educ_coef = s["coefficients"]["educ"]["coef"]
print(f"\n教育年限每增加 1 年，工资增加 {educ_coef*100:.2f}%")

# ── 可视化：残差分布 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

resid = model.residuals()
axes[0].hist(resid, bins=40, edgecolor="white", color="#4C72B0")
axes[0].set_title("残差分布", fontsize=13)
axes[0].set_xlabel("残差")
axes[0].set_ylabel("频数")

axes[1].scatter(model.predict(df[["educ", "exper", "exper2"]]), resid,
                alpha=0.3, s=10, color="#C44E52")
axes[1].axhline(0, color="k", linewidth=0.8)
axes[1].set_title("残差 vs 拟合值", fontsize=13)
axes[1].set_xlabel("拟合值")
axes[1].set_ylabel("残差")

plt.tight_layout()
plt.savefig("examples/econometrics/01_ols_residuals.png", dpi=150)
print("\n图表已保存至 examples/econometrics/01_ols_residuals.png")
