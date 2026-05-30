"""
示例 02：双重差分（DiD）—— 最低工资政策效果
=============================================
复现 Card & Krueger (1994) 最低工资研究设计：
    log_employment = α + β₁·treated + β₂·post + β₃·treated×post + ε

运行：
    python examples/econometrics/02_did_minimum_wage.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empirlab.traditional.did import DiD, did_data

# ── 生成模拟数据 ───────────────────────────────────────────────────────
np.random.seed(42)
df = did_data(n_states=40, n_periods=4, true_effect=-0.08)
print(f"数据规模：{len(df)} 行（{df['state'].nunique()} 州 × {df['year'].nunique()} 期）")
print(df.head(8).to_string(index=False))

# ── DiD 估计 ──────────────────────────────────────────────────────────
model = DiD()
result = model.fit(
    df,
    outcome="log_employment",
    treatment="treated",
    post="post",
    entity="state",
    time="year",
)
print("\n" + "="*60)
print("DiD 估计结果")
print("="*60)
print(result.summary_table().round(4))

s = result.summary()
theta = s["coefficients"]["treated_x_post"]["coef"]
p = s["coefficients"]["treated_x_post"]["p_value"]
print(f"\nDiD 系数（真值 = -0.08）：{theta:.4f}，p 值 = {p:.4f}")

# ── 平行趋势图 ────────────────────────────────────────────────────────
means = (
    df.groupby(["treated", "year"])["log_employment"]
    .mean()
    .reset_index()
)
treat_df = means[means["treated"] == 1]
ctrl_df  = means[means["treated"] == 0]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(treat_df["year"], treat_df["log_employment"],
        "o-", label="处理组（提高最低工资州）", color="#C44E52", linewidth=2)
ax.plot(ctrl_df["year"],  ctrl_df["log_employment"],
        "s--", label="对照组（未提高最低工资州）", color="#4C72B0", linewidth=2)
ax.axvline(x=df[df["post"] == 1]["year"].min() - 0.5,
           color="gray", linestyle=":", linewidth=1.5, label="政策实施")
ax.set_xlabel("年份")
ax.set_ylabel("对数就业人数")
ax.set_title("双重差分：平行趋势图")
ax.legend()
plt.tight_layout()
plt.savefig("examples/econometrics/02_did_parallel_trends.png", dpi=150)
print("图表已保存至 examples/econometrics/02_did_parallel_trends.png")
