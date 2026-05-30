/* ============================================================
   Stata 对应代码：因果森林（Causal Forest）异质处理效应
   实现方式：Stata 调用 Python econml 或 R grf
   ============================================================ */

clear all
set more off

* ── 生成数据 ────────────────────────────────────────────────
set seed 42
set obs 1000

gen x1 = rnormal(0,1)
gen x2 = rnormal(0,1)
gen x3 = rnormal(0,1)

* 倾向得分：P(D=1|X) = sigmoid(x1)
gen prop = 1 / (1 + exp(-x1))
gen d = runiform() < prop

* 真实 CATE = 1.0 + x2（异质效应）
gen cate_true = 1.0 + x2
gen y = cate_true * d + x3 + rnormal(0, 0.5)


/* ── 方法 1：Python econml（推荐）──────────────────────────────
   安装：pip install econml
   ──────────────────────────────────────────────────────────── */

python:
import numpy as np
import pandas as pd
from sfi import Data

# 读取 Stata 数据
n = Data.getObsTotal()
df = pd.DataFrame({
    "x1": Data.get("x1"),
    "x2": Data.get("x2"),
    "x3": Data.get("x3"),
    "d":  Data.get("d"),
    "y":  Data.get("y"),
})

X = df[["x1","x2","x3"]].to_numpy()
D = df["d"].to_numpy()
Y = df["y"].to_numpy()

try:
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100),
        model_t=GradientBoostingClassifier(n_estimators=100),
        n_estimators=200,
        random_state=42,
    )
    cf.fit(Y, D, X=X)

    cate = cf.effect(X)
    ate  = cate.mean()
    print(f"ATE（econml CausalForestDML）= {ate:.4f}（真值 = 1.0）")

    # 异质性检验：按 x2 分位数
    q_bins = np.percentile(df["x2"], [0, 25, 50, 75, 100])
    for i in range(4):
        mask = (df["x2"].values >= q_bins[i]) & (df["x2"].values < q_bins[i+1])
        print(f"  Q{i+1} (x2 ∈ [{q_bins[i]:.2f}, {q_bins[i+1]:.2f}]): CATE = {cate[mask].mean():.4f}")

except ImportError:
    # 回退到 T-Learner（无需 econml）
    print("econml 未安装，使用 T-Learner 替代...")
    from sklearn.ensemble import GradientBoostingRegressor

    mask1 = D == 1
    mask0 = D == 0
    m1 = GradientBoostingRegressor(n_estimators=200, random_state=42)
    m0 = GradientBoostingRegressor(n_estimators=200, random_state=42)
    m1.fit(X[mask1], Y[mask1])
    m0.fit(X[mask0], Y[mask0])
    cate = m1.predict(X) - m0.predict(X)
    print(f"ATE（T-Learner）= {cate.mean():.4f}（真值 = 1.0）")
end


/* ── T-Learner 纯 Stata 近似（调用 rforest）────────────────
   ssc install rforest
   ──────────────────────────────────────────────────────────── */

* 处理组
* preserve
* keep if d == 1
* rforest y x1-x3, type(reg) iter(200)
* predict mu1_treat
* restore
*
* preserve
* keep if d == 0
* rforest y x1-x3, type(reg) iter(200)
* predict mu0_ctrl
* restore
*
* 全样本预测（需手动合并）—— 略，建议使用 Python 路径
