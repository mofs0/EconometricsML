/* ============================================================
   Stata 对应代码：随机森林（通过 Python 调用）
   Stata 版本：16+（内置 Python 集成）
   注意：Stata 无内置 RF；本文件展示两种方法：
     1. Stata 16+ 调用 Python sklearn（推荐）
     2. 使用 randomforest 第三方包（ssc install randomforest）
   ============================================================ */

clear all
set more off

* ── 生成模拟数据 ────────────────────────────────────────────
set seed 42
set obs 500
forvalues i = 1/10 {
    gen x`i' = rnormal(0, 1)
}
gen y = 2*x1^2 - 1.5*x2*x3 + sin(x4) + rnormal(0, 0.5)


/* ── 方法 1：Stata 调用 Python sklearn（Stata 16+）──────────
   前提：已安装 Python + scikit-learn
   设置方法：python set exec "C:\...\python.exe"
   ──────────────────────────────────────────────────────────── */

python:
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sfi import Data

# 从 Stata 读取数据
varnames = ["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","y"]
n = Data.getObsTotal()
data = {v: Data.get(v) for v in varnames}
df = pd.DataFrame(data)

X = df.drop(columns="y")
y = df["y"]

# 训练随机森林
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X, y)
print(f"OOB R² = {rf.oob_score_:.4f}")

# 5折交叉验证
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
print(f"5折CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 变量重要性（Gini）
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance_sorted = importance.sort_values(ascending=False)
print("\n变量重要性（Gini，前5）：")
print(importance_sorted.head().round(4))

# 置换重要性（更可靠）
pi = permutation_importance(rf, X, y, n_repeats=20, random_state=42)
perm_imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
print("\n置换重要性（前5）：")
print(perm_imp.head().round(4))
end


/* ── 方法 2：第三方包 rforest（无需 Python）─────────────────
   安装：ssc install rforest
   注意：功能较 sklearn 有限，但不需要 Python 环境
   ──────────────────────────────────────────────────────────── */

* rforest y x1-x10, type(reg) numvars(3) iter(200)
* predict yhat_rf
* gen resid_rf = y - yhat_rf
* sum resid_rf
