"""事件研究法 (Event Study)
============================
来源参考：MacKinlay (1997). "Event Studies in Economics and Finance". JEL;
          陈超, 何佳 (2019). "股权质押、控制权转移风险与股价崩盘".
          《管理世界》第35卷第2期.
适用场景：检验事件（政策公告、并购、盈利公告）对资产价格的超常影响。
Python 版本：3.10+
依赖：numpy >= 1.24, pandas >= 1.5, scipy >= 1.10
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


class EventStudy:
    """市场模型事件研究法.

    步骤：
    1. 估计期（estimation window）：OLS 估计市场模型 r_i = α + β·r_m + ε
    2. 事件期（event window）：计算超常收益 AR = r_i - (α_hat + β_hat·r_m)
    3. 累计超常收益 CAR = ΣAR，检验显著性

    参数
    ----
    est_start  : int  估计期起始（相对事件日，如 -250）
    est_end    : int  估计期结束（如 -11）
    event_start: int  事件窗起始（如 -5）
    event_end  : int  事件窗结束（如 5）

    示例
    ----
    >>> from empirlab.traditional.event_study import EventStudy, stock_event_data
    >>> ret_firm, ret_mkt = stock_event_data()
    >>> m = EventStudy(est_start=-200, est_end=-11, event_start=-5, event_end=5)
    >>> m.fit(ret_firm, ret_mkt)
    >>> print(m.summary_table().round(4))
    """

    def __init__(self, est_start: int = -250, est_end: int = -11,
                 event_start: int = -5, event_end: int = 5):
        self.est_start = est_start
        self.est_end = est_end
        self.event_start = event_start
        self.event_end = event_end
        self.is_fitted = False
        self._alpha: float = 0.0
        self._beta: float = 1.0
        self._sigma2: float = 0.0
        self._AR: Optional[np.ndarray] = None
        self._CAR: Optional[float] = None

    def _market_model(self, r_firm, r_mkt):
        """OLS 估计市场模型参数。"""
        X = np.column_stack([np.ones_like(r_mkt), r_mkt])
        beta = np.linalg.pinv(X) @ r_firm
        resid = r_firm - X @ beta
        sigma2 = float(np.var(resid, ddof=2))
        return float(beta[0]), float(beta[1]), sigma2

    def fit(self, returns_firm: np.ndarray,
            returns_market: np.ndarray,
            event_date_idx: Optional[int] = None) -> "EventStudy":
        """拟合市场模型并计算超常收益。

        参数
        ----
        returns_firm   : 个股日收益率序列（对齐到事件日 t=0）
        returns_market : 市场指数日收益率序列（同长度）
        event_date_idx : 事件日在序列中的位置（None 时默认取中间）
        """
        r_f = np.asarray(returns_firm, dtype=float)
        r_m = np.asarray(returns_market, dtype=float)
        n = len(r_f)
        if event_date_idx is None:
            event_date_idx = n // 2

        # 将序列映射到相对日期
        t = np.arange(n) - event_date_idx  # t=0 为事件日

        est_mask = (t >= self.est_start) & (t <= self.est_end)
        evt_mask = (t >= self.event_start) & (t <= self.event_end)

        if est_mask.sum() < 20:
            raise ValueError(f"估计期样本不足（{est_mask.sum()} 天），请调整窗口。")

        alpha, beta, sigma2 = self._market_model(r_f[est_mask], r_m[est_mask])
        self._alpha, self._beta, self._sigma2 = alpha, beta, sigma2

        AR_evt = r_f[evt_mask] - (alpha + beta * r_m[evt_mask])
        self._AR = AR_evt
        self._t_evt = t[evt_mask]
        self._CAR = float(np.sum(AR_evt))
        self._n_est = int(est_mask.sum())
        self._n_evt = int(evt_mask.sum())
        self.is_fitted = True
        return self

    def _t_test_AR(self) -> Tuple[np.ndarray, np.ndarray]:
        """每日 AR 的 t 检验（Patell 标准化）。"""
        se = np.sqrt(self._sigma2)
        t_stats = self._AR / se
        p_vals = 2 * stats.t.sf(np.abs(t_stats), df=self._n_est - 2)
        return t_stats, p_vals

    def _t_test_CAR(self) -> Tuple[float, float]:
        """CAR 的联合 t 检验。"""
        se_car = np.sqrt(self._n_evt * self._sigma2)
        t_stat = self._CAR / se_car
        p_val = float(2 * stats.t.sf(abs(t_stat), df=self._n_est - 2))
        return float(t_stat), p_val

    def summary(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        t_ar, p_ar = self._t_test_AR()
        t_car, p_car = self._t_test_CAR()
        ar_table = {
            int(self._t_evt[i]): dict(
                AR=float(self._AR[i]),
                t=float(t_ar[i]),
                p_value=float(p_ar[i]),
            )
            for i in range(len(self._AR))
        }
        return {
            "model": "EventStudy-MarketModel",
            "n_estimation": self._n_est,
            "n_event_window": self._n_evt,
            "market_model": {"alpha": self._alpha, "beta": self._beta, "sigma2": self._sigma2},
            "AR_by_day": ar_table,
            "CAR": {
                "coef": self._CAR,
                "t": t_car,
                "p_value": p_car,
            },
        }

    def summary_table(self) -> pd.DataFrame:
        s = self.summary()
        rows = [{"t_day": t, **v} for t, v in s["AR_by_day"].items()]
        df = pd.DataFrame(rows).set_index("t_day")
        df.loc["CAR", "AR"] = s["CAR"]["coef"]
        df.loc["CAR", "t"]  = s["CAR"]["t"]
        df.loc["CAR", "p_value"] = s["CAR"]["p_value"]
        return df


def stock_event_data(n_days: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """生成单只股票事件研究合成数据（事件日 = 序列中点，t=0）。

    DGP：市场模型 + 事件窗 [-3, 3] 内存在显著 AR（如 M&A 公告）。
    """
    rng = np.random.default_rng(seed)
    r_mkt = rng.normal(0.0005, 0.012, n_days)
    alpha, beta = 0.0002, 1.1
    r_firm = alpha + beta * r_mkt + rng.normal(0, 0.015, n_days)
    event_idx = n_days // 2
    # 事件窗 [-3, 3] 注入 +0.01 超常收益（正面公告）
    for d in range(event_idx - 3, event_idx + 4):
        r_firm[d] += 0.01
    return r_firm, r_mkt


if __name__ == "__main__":
    import pprint
    r_firm, r_mkt = stock_event_data()
    m = EventStudy(est_start=-200, est_end=-11, event_start=-5, event_end=5)
    m.fit(r_firm, r_mkt)
    pprint.pprint({k: v for k, v in m.summary().items() if k != 'AR_by_day'})
    print(); print(m.summary_table().round(4))
