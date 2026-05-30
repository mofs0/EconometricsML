"""
示例 06：因果森林 —— 异质处理效应（HTE）估计
=============================================
场景：政府补贴对不同规模企业的研发投入影响是否存在异质性？
方法：T-Learner Meta-Learner（因果森林近似实现）

研究发现：小企业（x2 高）可能比大企业（x2 低）受益更多

运行：
    python examples/econometrics/06_causal_forest_hte.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empirlab.ml.causal_forest import CausalForest, cf_data

# ── 数据（含异质效应）────────────────────────────────────────────────
TRUE_ATE = 1.0
df = cf_data(n=1000, true_ate=TRUE_ATE, heterogeneous=True, seed=42)
print(f"样本量：{len(df)}, 处理率：{df['d'].mean():.2%}")

# ── T-Learner 估计 ────────────────────────────────────────────────────
model = CausalForest(learner="T")
model.fit(df[["x1", "x2", "x3"]], df["d"], df["y"], n_bootstrap=200)

s = model.summary()
print(f"\nATE = {s['ATE']:.4f}（真值 = {TRUE_ATE}）")
print(f"SE  = {s['SE']:.4f}")
print(f"95% CI：[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]")
print(f"p 值 = {s['p_value']:.4f}")

print("\nCATE 分布：")
print(model.cate_summary().to_string(index=False))

# ── 异质性可视化 ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# CATE 分布直方图
axes[0].hist(model.cate_, bins=40, color="#4C72B0", edgecolor="white")
axes[0].axvline(model.ate_, color="#C44E52", linewidth=2,
                label=f"ATE = {model.ate_:.3f}")
axes[0].set_xlabel("CATE 估计值")
axes[0].set_ylabel("频数")
axes[0].set_title("CATE 分布")
axes[0].legend()

# CATE vs x2（异质性来源）
axes[1].scatter(df["x2"], model.cate_, alpha=0.15, s=8, color="#4C72B0")
# 拟合趋势线
from numpy.polynomial.polynomial import polyfit
c = polyfit(df["x2"].values, model.cate_, 1)
x_range = np.linspace(df["x2"].min(), df["x2"].max(), 100)
axes[1].plot(x_range, c[0] + c[1] * x_range, color="#C44E52", linewidth=2)
axes[1].set_xlabel("x2（企业规模代理）")
axes[1].set_ylabel("CATE 估计值")
axes[1].set_title("CATE vs 规模（异质性检验）")

# CATE 按分位数分组的平均值
quantiles = pd.qcut(df["x2"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
cate_by_q = pd.Series(model.cate_).groupby(quantiles).mean()
axes[2].bar(cate_by_q.index, cate_by_q.values, color="#4C72B0", edgecolor="white")
axes[2].axhline(model.ate_, color="#C44E52", linewidth=2, linestyle="--",
                label=f"ATE = {model.ate_:.3f}")
axes[2].set_xlabel("x2 五分位组")
axes[2].set_ylabel("平均 CATE")
axes[2].set_title("各分位组平均处理效应")
axes[2].legend()

plt.suptitle("因果森林：异质处理效应分析", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("examples/econometrics/06_causal_forest_hte.png", dpi=150, bbox_inches="tight")
print("图表已保存至 examples/econometrics/06_causal_forest_hte.png")
