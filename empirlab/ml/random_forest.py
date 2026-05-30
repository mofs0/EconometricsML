"""随机森林回归（经济学应用版）
================================
来源参考：Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
          Athey, S., & Imbens, G. (2019). Machine learning methods that economists
          should know about. *Annual Review of Economics*, 11, 685–725.

适用场景：
  - 非线性控制函数估计（作为 nuisance 函数）
  - 变量重要性筛选
  - 预测型研究（预测企业绩效、违约率等）
  - Double ML 中的 g_hat / m_hat

说明：
  本模块在 sklearn RandomForestRegressor 基础上封装了
  ① 变量重要性可视化表
  ② OOB（Out-of-Bag）误差报告
  ③ 置换重要性（Permutation Importance，更可靠）
  供计量经济学 notebook 直接调用。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as _RF
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class RFRegressor:
    """随机森林回归封装（经济学研究用途）。

    参数
    ----
    n_estimators : int, default 300
    max_depth    : int or None, default None（完全生长）
    min_samples_leaf : int, default 5
    random_state : int, default 42
    n_jobs       : int, default -1（使用全部 CPU）

    属性
    ----
    oob_score_       : float —— 袋外 R²
    feature_names_   : list[str]
    importance_table_: pd.DataFrame —— 变量重要性

    示例
    ----
    >>> from empirlab.ml import RFRegressor, rf_data
    >>> df = rf_data(n=500)
    >>> y = df.pop("y")
    >>> model = RFRegressor().fit(df, y)
    >>> print(model.importance_table_.head())
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._model: Optional[_RF] = None
        self.oob_score_: Optional[float] = None
        self.feature_names_: Optional[List[str]] = None
        self.importance_table_: Optional[pd.DataFrame] = None
        self._X_fit: Optional[np.ndarray] = None
        self._y_fit: Optional[np.ndarray] = None

    def fit(self, X, y, compute_permutation: bool = False) -> "RFRegressor":
        """拟合随机森林。

        参数
        ----
        X                    : pd.DataFrame 或 array-like
        y                    : array-like
        compute_permutation  : bool —— 是否计算置换重要性（更慢但更可靠）
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_names_ = [f"x{i+1}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=float).ravel()
        self._X_fit = X_arr
        self._y_fit = y_arr

        self._model = _RF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X_arr, y_arr)
        self.oob_score_ = float(self._model.oob_score_)

        # 基于不纯度的重要性
        imp_gini = self._model.feature_importances_

        rows = [{"variable": nm, "importance_gini": float(ig)}
                for nm, ig in zip(self.feature_names_, imp_gini)]

        if compute_permutation:
            pi = permutation_importance(
                self._model, X_arr, y_arr,
                n_repeats=20, random_state=self.random_state, n_jobs=self.n_jobs
            )
            for i, row in enumerate(rows):
                row["importance_perm_mean"] = float(pi.importances_mean[i])
                row["importance_perm_std"] = float(pi.importances_std[i])

        self.importance_table_ = (
            pd.DataFrame(rows)
            .sort_values("importance_gini", ascending=False)
            .reset_index(drop=True)
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("请先调用 fit()。")
        X_arr = np.asarray(X, dtype=float)
        return self._model.predict(X_arr)

    def cv_score(self, X, y, cv: int = 5) -> dict:
        """5折交叉验证 R²。"""
        if self._model is None:
            raise RuntimeError("请先调用 fit()。")
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        scores = cross_val_score(self._model, X_arr, y_arr, cv=cv, scoring="r2")
        return {"cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std())}

    def summary(self) -> dict:
        """返回模型评估摘要。"""
        if self._model is None:
            raise RuntimeError("请先调用 fit()。")
        return {
            "n_estimators": self.n_estimators,
            "oob_r2": self.oob_score_,
            "n_features": len(self.feature_names_),
            "top5_vars": list(self.importance_table_["variable"].head(5)),
        }


# ── 辅助数据生成 ──────────────────────────────────────────────────────

def rf_data(n: int = 500, p: int = 10, seed: int = 42) -> pd.DataFrame:
    """生成含非线性关系的合成数据。

    DGP：y = 2*x1^2 - 1.5*x2*x3 + sin(x4) + noise
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (
        2.0 * X[:, 0] ** 2
        - 1.5 * X[:, 1] * X[:, 2]
        + np.sin(X[:, 3])
        + rng.normal(0, 0.5, n)
    )
    cols = {f"x{i+1}": X[:, i] for i in range(p)}
    cols["y"] = y
    return pd.DataFrame(cols)


if __name__ == "__main__":
    df = rf_data(n=800)
    y = df["y"]
    X = df.drop(columns="y")

    model = RFRegressor(n_estimators=200).fit(X, y, compute_permutation=True)
    print("模型摘要：", model.summary())
    print("\n变量重要性（前5）：")
    print(model.importance_table_.head().round(4))
    cv = model.cv_score(X, y)
    print(f"\n5折CV R² = {cv['cv_r2_mean']:.4f} ± {cv['cv_r2_std']:.4f}")
