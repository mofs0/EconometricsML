"""倾向得分匹配 (Propensity Score Matching)
============================================
来源参考：Rosenbaum & Rubin (1983). "The Central Role of the Propensity Score". Biometrika;
          余明桂, 回雅甫, 潘红波 (2010). "政治联系、寻租与地方政府财政补贴有效性".
          《经济研究》第3期.
适用场景：观测数据中控制选择偏差，估计平均处理效应（ATT）。
Python 版本：3.10+
依赖：numpy >= 1.24, pandas >= 1.5, scipy >= 1.10
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from scipy import optimize, stats

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class PSM:
    """倾向得分匹配估计量 (Propensity Score Matching).

    流程：
    1. Logit 估计倾向得分 P(D=1|X)
    2. 最近邻匹配（1:k，含/不含放回，可设 caliper）
    3. 计算 ATT = E[Y(1)-Y(0)|D=1]

    参数
    ----
    n_neighbors : int, default 1  每个处理组单元匹配的控制组数量
    caliper     : float or None   最大允许倾向得分距离（None=不限）
    replacement : bool, default False  是否放回匹配
    random_state: int, default 42

    示例
    ----
    >>> from empirlab.traditional.psm import PSM, subsidy_data
    >>> df = subsidy_data()
    >>> m = PSM(n_neighbors=1, caliper=0.05)
    >>> m.fit(df[['lnsize','lnage','lev','roa']], df['subsidy'], df['lnrd'])
    >>> print(m.summary())
    """

    def __init__(self, n_neighbors: int = 1,
                 caliper: Optional[float] = None,
                 replacement: bool = False,
                 random_state: int = 42):
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.replacement = replacement
        self.random_state = random_state
        self.is_fitted = False
        self._ps: Optional[np.ndarray] = None
        self._matched_idx: Optional[np.ndarray] = None
        self._ATT: Optional[float] = None
        self._ATT_se: Optional[float] = None

    def _logit_ps(self, X, D):
        """用 Logit MLE 估计倾向得分。"""
        from scipy.special import expit
        ones = np.ones((X.shape[0], 1))
        Xd = np.hstack([ones, X])
        def neg_ll(b):
            p = np.clip(expit(Xd @ b), 1e-12, 1-1e-12)
            return -np.sum(D*np.log(p) + (1-D)*np.log(1-p))
        def grad(b):
            p = expit(Xd @ b)
            return -Xd.T @ (D - p)
        res = optimize.minimize(neg_ll, np.zeros(Xd.shape[1]), jac=grad, method='BFGS')
        return expit(Xd @ res.x)

    def fit(self, X, treatment, outcome) -> "PSM":
        """执行 PSM 并估计 ATT。

        参数
        ----
        X         : array-like or DataFrame，协变量矩阵
        treatment : array-like，处理变量（0/1）
        outcome   : array-like，结果变量
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self._feature_names = None
        D = np.asarray(treatment, dtype=float).ravel()
        y = np.asarray(outcome, dtype=float).ravel()

        ps = self._logit_ps(X_arr, D)
        self._ps = ps

        treat_idx = np.where(D == 1)[0]
        ctrl_idx  = np.where(D == 0)[0]

        rng = np.random.default_rng(self.random_state)
        matched_ctrl = []
        used_ctrl = [] if not self.replacement else None

        for i in treat_idx:
            dists = np.abs(ps[ctrl_idx] - ps[i])
            if self.caliper is not None:
                valid = ctrl_idx[dists <= self.caliper]
                if len(valid) == 0:
                    matched_ctrl.append(None)
                    continue
                dists_valid = np.abs(ps[valid] - ps[i])
            else:
                valid = ctrl_idx
                dists_valid = dists
            if not self.replacement and used_ctrl is not None:
                avail = [c for c in valid if c not in used_ctrl]
                if len(avail) == 0:
                    matched_ctrl.append(None)
                    continue
                dists_v2 = np.abs(ps[avail] - ps[i])
                k = min(self.n_neighbors, len(avail))
                chosen = np.array(avail)[np.argsort(dists_v2)[:k]]
            else:
                k = min(self.n_neighbors, len(valid))
                chosen = valid[np.argsort(dists_valid)[:k]]
            if used_ctrl is not None:
                used_ctrl.extend(chosen.tolist())
            matched_ctrl.append(chosen)

        valid_mask = [c is not None for c in matched_ctrl]
        treat_valid = treat_idx[valid_mask]
        ctrl_valid  = [c for c in matched_ctrl if c is not None]

        y_treat = y[treat_valid]
        y_ctrl  = np.array([y[c].mean() for c in ctrl_valid])
        diffs = y_treat - y_ctrl

        self._ATT = float(np.mean(diffs))
        self._ATT_se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        self._n_treated = len(treat_valid)
        self._n_matched = len(ctrl_valid)
        self._n_unmatched = int(np.sum(~np.array(valid_mask)))
        self.is_fitted = True
        return self

    def summary(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        t = self._ATT / self._ATT_se
        p = float(2 * stats.t.sf(abs(t), df=self._n_treated - 1))
        cv = stats.t.ppf(0.975, df=self._n_treated - 1)
        return {
            "model": "PSM",
            "n_neighbors": self.n_neighbors,
            "caliper": self.caliper,
            "replacement": self.replacement,
            "n_treated": self._n_treated,
            "n_matched": self._n_matched,
            "n_unmatched": self._n_unmatched,
            "ATT": {
                "coef": self._ATT,
                "se": self._ATT_se,
                "t": float(t),
                "p_value": p,
                "ci_lower": float(self._ATT - cv * self._ATT_se),
                "ci_upper": float(self._ATT + cv * self._ATT_se),
            },
            "ps_stats": {
                "mean_treated": float(self._ps[self._ps >= 0.5].mean()),
                "mean_control": float(self._ps[self._ps < 0.5].mean()),
            },
        }

    def summary_table(self) -> pd.DataFrame:
        s = self.summary()
        return pd.DataFrame([{"parameter": "ATT", **s["ATT"]}]).set_index("parameter")


def subsidy_data(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """余明桂等 (2010) 风格：政府补贴对企业研发投入的影响。

    DGP：
        subsidy*  = -2 + 0.5*lnsize + 0.3*lev - 0.4*roa + eps_d
        subsidy   = 1{subsidy* > 0}
        lnrd = 0.5 + 0.3*subsidy + 0.4*lnsize + 0.2*lnprod + eta
    """
    rng = np.random.default_rng(seed)
    lnsize = rng.normal(4.0, 1.0, n)
    lnage  = rng.uniform(1.0, 4.0, n)
    lev    = rng.beta(2, 5, n)
    roa    = rng.normal(0.05, 0.06, n)
    xb_d   = -2.0 + 0.5*lnsize + 0.3*lev - 0.4*roa + rng.normal(0,1,n)
    from scipy.special import expit
    subsidy = (rng.uniform(size=n) < expit(xb_d)).astype(float)
    lnrd   = 0.5 + 0.3*subsidy + 0.4*lnsize + 0.2*rng.normal(3,0.5,n) + rng.normal(0,0.4,n)
    return pd.DataFrame(dict(lnsize=lnsize, lnage=lnage, lev=lev, roa=roa,
                             subsidy=subsidy, lnrd=lnrd))


if __name__ == "__main__":
    import pprint
    df = subsidy_data()
    m = PSM(n_neighbors=1, caliper=0.05)
    m.fit(df[['lnsize','lnage','lev','roa']], df['subsidy'], df['lnrd'])
    pprint.pprint(m.summary())
    print(); print(m.summary_table().round(4))
