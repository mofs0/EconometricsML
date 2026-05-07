"""面板数据回归：固定效应 / 随机效应 / Hausman 检验
======================================================
来源参考：Wooldridge (2010) Ch.10-11;
          鲁晓东 & 连玉君 (2012). "中国工业企业全要素生产率估算". 《经济研究》第2期.
适用场景：面板数据（个体 × 时间）的 FE/RE 估计与模型选择。
Python 版本：3.10+
依赖：numpy >= 1.24, pandas >= 1.5, scipy >= 1.10
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from scipy import stats

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
from empirlab.utils.metrics import calculate_metrics


class PanelOLS:
    """面板数据 OLS：固定效应（Within）或随机效应（GLS）。

    参数
    ----
    effect : {'fixed', 'random'}, default 'fixed'
    entity_col : str  个体列名（默认 'id'）
    time_col   : str  时间列名（默认 'year'）
    robust : bool, default False  HC1 稳健标准误（FE 下有效）

    示例
    ----
    >>> from empirlab.traditional.panel import PanelOLS, firm_panel_data
    >>> df = firm_panel_data()
    >>> m = PanelOLS(effect='fixed', robust=True)
    >>> m.fit(df, ['lnK','lnL','export'], 'lnY', entity_col='firm_id', time_col='year')
    >>> print(m.summary_table().round(4))
    """

    def __init__(self, effect: Literal['fixed','random'] = 'fixed',
                 robust: bool = False):
        self.effect = effect
        self.robust = robust
        self.is_fitted = False
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self._resid: Optional[np.ndarray] = None
        self._y_demeaned: Optional[np.ndarray] = None
        self._X_demeaned: Optional[np.ndarray] = None
        self._n: int = 0
        self._T: float = 0.0
        self._N_obs: int = 0

    def _within_demean(self, df, y_col, x_cols, entity_col):
        """组内去均值（固定效应变换）。"""
        df2 = df.copy()
        for col in x_cols + [y_col]:
            df2[f'_mean_{col}'] = df2.groupby(entity_col)[col].transform('mean')
            df2[f'_dm_{col}'] = df2[col] - df2[f'_mean_{col}']
        y_dm = df2[f'_dm_{y_col}'].to_numpy(dtype=float)
        X_dm = df2[[f'_dm_{c}' for c in x_cols]].to_numpy(dtype=float)
        return X_dm, y_dm

    def _re_transform(self, df, y_col, x_cols, entity_col):
        """随机效应准去均值变换（Mundlak/Swamy-Arora）。"""
        df2 = df.copy()
        n_entities = df2[entity_col].nunique()
        T_avg = len(df2) / n_entities
        # 简化：用 OLS 残差估计 sigma_u
        X_fe, y_fe = self._within_demean(df2, y_col, x_cols, entity_col)
        b_fe = np.linalg.pinv(X_fe) @ y_fe
        res_fe = y_fe - X_fe @ b_fe
        sigma_e2 = np.sum(res_fe**2) / (len(y_fe) - n_entities - len(x_cols))
        # 个体均值
        for col in x_cols + [y_col]:
            df2[f'_mean_{col}'] = df2.groupby(entity_col)[col].transform('mean')
        # 方差分解：估算 sigma_u2
        u_hat = df2.groupby(entity_col)[y_col].mean().to_numpy() - \
                df2.groupby(entity_col)[[f'_mean_{c}' for c in x_cols]].mean().to_numpy() @ b_fe
        sigma_u2 = max(float(np.var(u_hat)) - sigma_e2 / T_avg, 1e-10)
        theta = 1 - np.sqrt(sigma_e2 / (T_avg * sigma_u2 + sigma_e2))
        # 准去均值
        for col in x_cols + [y_col]:
            df2[f'_re_{col}'] = df2[col] - theta * df2[f'_mean_{col}']
        y_re = df2[f'_re_{y_col}'].to_numpy(dtype=float)
        X_re = df2[[f'_re_{c}' for c in x_cols]].to_numpy(dtype=float)
        const_re = (1 - theta) * np.ones((len(y_re), 1))
        X_re_full = np.hstack([const_re, X_re])
        return X_re_full, y_re, theta

    def fit(self, df: pd.DataFrame, x_cols: List[str], y_col: str,
            entity_col: str = 'id', time_col: str = 'year') -> "PanelOLS":
        """拟合面板模型。

        参数
        ----
        df         : 长格式 DataFrame，必须含 entity_col 和 time_col
        x_cols     : 自变量列名列表
        y_col      : 因变量列名
        entity_col : 个体 ID 列
        time_col   : 时间列
        """
        self.feature_names_ = x_cols
        self._n = df[entity_col].nunique()
        self._T = len(df) / self._n
        self._N_obs = len(df)

        if self.effect == 'fixed':
            X_dm, y_dm = self._within_demean(df, y_col, x_cols, entity_col)
            self.coef_ = np.linalg.pinv(X_dm) @ y_dm
            self._X_demeaned = X_dm
            self._y_demeaned = y_dm
            self._resid = y_dm - X_dm @ self.coef_
        else:
            X_re, y_re, _ = self._re_transform(df, y_col, x_cols, entity_col)
            beta = np.linalg.pinv(X_re) @ y_re
            self.coef_ = beta[1:]
            self._intercept = float(beta[0])
            self._X_demeaned = X_re[:, 1:]
            self._y_demeaned = y_re
            self._resid = y_re - X_re @ beta

        self.is_fitted = True
        return self

    def _se(self) -> np.ndarray:
        n_obs = self._N_obs
        k = len(self.coef_)
        df_r = n_obs - k - self._n if self.effect == 'fixed' else n_obs - k - 1
        df_r = max(df_r, 1)
        X = self._X_demeaned
        resid = self._resid
        if self.robust:
            XtX_inv = np.linalg.pinv(X.T @ X)
            Xe = X * resid[:, None]
            meat = Xe.T @ Xe
            cov = (n_obs / df_r) * XtX_inv @ meat @ XtX_inv
        else:
            sigma2 = np.sum(resid**2) / df_r
            cov = sigma2 * np.linalg.pinv(X.T @ X)
        return np.sqrt(np.diag(cov))

    def summary(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()。")
        se = self._se()
        t = self.coef_ / se
        k = len(self.coef_)
        df_r = max(self._N_obs - k - (self._n if self.effect == 'fixed' else 1), 1)
        p = 2 * stats.t.sf(np.abs(t), df=df_r)
        cv = stats.t.ppf(0.975, df=df_r)
        r2 = 1 - np.sum(self._resid**2) / np.sum((self._y_demeaned - self._y_demeaned.mean())**2)
        coef_tbl = {
            nm: dict(coef=float(self.coef_[i]), se=float(se[i]),
                     t=float(t[i]), p_value=float(p[i]),
                     ci_lower=float(self.coef_[i] - cv * se[i]),
                     ci_upper=float(self.coef_[i] + cv * se[i]))
            for i, nm in enumerate(self.feature_names_)
        }
        return {
            "model": f"Panel-{self.effect.upper()}",
            "se_type": "HC1" if self.robust else "OLS",
            "n_obs": self._N_obs, "n_entities": self._n,
            "avg_T": round(self._T, 1),
            "coefficients": coef_tbl,
            "fit": {"within_R2": float(r2)},
            "note": "FE 系数不含时间均值，intercept 被 demean 吸收。"
        }

    def summary_table(self) -> pd.DataFrame:
        s = self.summary()
        rows = [{"parameter": nm, **vals} for nm, vals in s["coefficients"].items()]
        return pd.DataFrame(rows).set_index("parameter")

    @staticmethod
    def hausman_test(fe_model: "PanelOLS", re_model: "PanelOLS") -> dict:
        """Hausman 检验：FE vs RE。H0：RE 一致（随机效应适合）。"""
        b_fe = fe_model.coef_
        b_re = re_model.coef_
        diff = b_fe - b_re
        var_fe = np.diag(np.linalg.pinv(fe_model._X_demeaned.T @ fe_model._X_demeaned) *
                         (np.sum(fe_model._resid**2) / max(fe_model._N_obs - len(b_fe) - fe_model._n, 1)))
        var_re = np.diag(np.linalg.pinv(re_model._X_demeaned.T @ re_model._X_demeaned) *
                         (np.sum(re_model._resid**2) / max(re_model._N_obs - len(b_re) - 1, 1)))
        var_diff = np.diag(np.abs(np.diag(np.diag(var_fe) - np.diag(var_re))))
        k = len(diff)
        chi2 = float(diff @ np.linalg.pinv(var_diff) @ diff)
        pval = float(stats.chi2.sf(chi2, df=k))
        return {"H_stat": chi2, "df": k, "p_value": pval,
                "conclusion": "使用 FE（拒绝 RE）" if pval < 0.05 else "使用 RE（不拒绝）"}


def firm_panel_data(n_firms: int = 100, n_years: int = 8, seed: int = 42) -> pd.DataFrame:
    """生成企业面板数据：Cobb-Douglas 生产函数。

    DGP（鲁晓东 & 连玉君 2012 风格）：
        lnY = 0.4*lnK + 0.5*lnL + 0.2*export + alpha_i + eps
    """
    rng = np.random.default_rng(seed)
    firm_ids = np.repeat(np.arange(1, n_firms + 1), n_years)
    years = np.tile(np.arange(2010, 2010 + n_years), n_firms)
    alpha_i = rng.normal(0, 0.5, n_firms)
    alpha_i_rep = np.repeat(alpha_i, n_years)
    lnK    = rng.normal(4.0, 1.0, n_firms * n_years)
    lnL    = rng.normal(3.5, 0.8, n_firms * n_years)
    export = rng.binomial(1, 0.3, n_firms * n_years).astype(float)
    eps    = rng.normal(0, 0.3, n_firms * n_years)
    lnY    = 0.4 * lnK + 0.5 * lnL + 0.2 * export + alpha_i_rep + eps
    return pd.DataFrame(dict(firm_id=firm_ids, year=years, lnY=lnY, lnK=lnK, lnL=lnL, export=export))


if __name__ == "__main__":
    import pprint
    df = firm_panel_data()
    fe = PanelOLS(effect='fixed', robust=True)
    fe.fit(df, ['lnK','lnL','export'], 'lnY', entity_col='firm_id', time_col='year')
    print("=== 固定效应 ===")
    pprint.pprint(fe.summary())
    print(); print(fe.summary_table().round(4))
