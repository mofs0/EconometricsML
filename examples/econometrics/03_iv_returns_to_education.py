"""
示例 03：工具变量（IV/2SLS）—— 教育回报率的因果识别
=====================================================
研究问题：教育对收入的因果效应
内生性来源：能力偏误（高能力者既受教育多，又收入高）
工具变量策略：地理距离到最近大学（Card 1995）

运行：
    python examples/econometrics/03_iv_returns_to_education.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from empirlab.traditional.iv import IV2SLS, iv_data
from empirlab.traditional.ols import OLS

# ── 生成数据 ──────────────────────────────────────────────────────────
df = iv_data(n=800, seed=42)
print(f"样本量 N = {len(df)}")
print(df.describe().round(3))

# ── OLS（偏误参考）────────────────────────────────────────────────────
ols = OLS(robust=True).fit(df[["educ", "exper"]], df["log_wage"])
ols_s = ols.summary()
print("\n" + "="*60)
print("OLS 估计（存在能力偏误）")
print("="*60)
print(ols.summary_table().round(4))

# ── IV / 2SLS ─────────────────────────────────────────────────────────
iv_model = IV2SLS(robust=True)
iv_result = iv_model.fit(
    endog_df=df[["log_wage"]],
    X=df[["exper"]],
    D=df[["educ"]],
    Z=df[["distance"]],
)
iv_s = iv_result.summary()
print("\n" + "="*60)
print("IV/2SLS 估计（工具变量：到最近大学的距离）")
print("="*60)
print(iv_result.summary_table().round(4))

# ── 对比 ──────────────────────────────────────────────────────────────
ols_educ = ols_s["coefficients"]["educ"]["coef"]
iv_educ  = iv_s["coefficients"]["educ"]["coef"]
print(f"\n教育回报率对比：")
print(f"  OLS（偏误）：{ols_educ*100:.2f}%/年")
print(f"  IV（因果）：{iv_educ*100:.2f}%/年")
print(f"\n一阶段 F 统计量：{iv_s.get('first_stage_F', 'N/A'):.2f}")
print("（F > 10 视为强工具变量）")
