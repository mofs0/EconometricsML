"""
Extended econometric models.

These implementations are lightweight but real, so they can be imported and
used directly from the package without changing the existing public API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


def _as_2d(x):
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class IVRegression:
    """Two-stage least squares / instrumental variables regression."""

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.fitted_values_ = None
        self.residuals_ = None

    def fit(self, y, exog, endog, instruments):
        y = np.asarray(y).reshape(-1)
        exog = _as_2d(exog)
        endog = _as_2d(endog)
        instruments = _as_2d(instruments)

        if self.fit_intercept:
            exog = np.column_stack([np.ones(len(y)), exog])
            instruments = np.column_stack([np.ones(len(y)), instruments])

        x = np.column_stack([exog, endog])
        pz = instruments @ np.linalg.pinv(instruments.T @ instruments) @ instruments.T
        beta = np.linalg.pinv(x.T @ pz @ x) @ (x.T @ pz @ y)

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self.fitted_values_ = x @ beta
        self.residuals_ = y - self.fitted_values_
        return self

    def predict(self, exog, endog):
        exog = _as_2d(exog)
        endog = _as_2d(endog)
        if self.fit_intercept:
            exog = np.column_stack([np.ones(len(exog)), exog])
        x = np.column_stack([exog, endog])
        beta = np.concatenate([[self.intercept_], self.coef_]) if self.fit_intercept else self.coef_
        return x @ beta


class FixedEffectsRegression:
    """Entity fixed effects via within transformation."""

    def __init__(self, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.residuals_ = None
        self.groups_ = None

    def fit(self, y, exog, groups):
        y = np.asarray(y).reshape(-1)
        x = _as_2d(exog)
        groups = np.asarray(groups)
        self.groups_ = groups

        y_dm = y.copy().astype(float)
        x_dm = x.copy().astype(float)
        for g in np.unique(groups):
            idx = groups == g
            y_dm[idx] -= y[idx].mean()
            x_dm[idx] -= x[idx].mean(axis=0)

        beta = np.linalg.pinv(x_dm.T @ x_dm) @ (x_dm.T @ y_dm)
        self.coef_ = beta
        self.residuals_ = y_dm - x_dm @ beta
        return self

    def predict(self, exog):
        return _as_2d(exog) @ self.coef_ + self.intercept_


class RandomEffectsRegression:
    """Random effects via statsmodels MixedLM."""

    def __init__(self):
        self.result_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, y, exog, groups):
        y = np.asarray(y).reshape(-1)
        x = _as_2d(exog)
        x = sm.add_constant(x, has_constant="add")
        self.result_ = MixedLM(y, x, groups=np.asarray(groups)).fit(reml=False, disp=False)
        self.intercept_ = float(self.result_.params[0])
        self.coef_ = np.asarray(self.result_.params[1:])
        return self

    def predict(self, exog):
        x = sm.add_constant(_as_2d(exog), has_constant="add")
        return self.result_.predict(exog=x)


class DifferenceInDifferences:
    """Simple DiD model with treatment, post, and interaction term."""

    def __init__(self):
        self.result_ = None
        self.treatment_effect_ = None

    def fit(self, y, treated, post, controls=None):
        y = np.asarray(y).reshape(-1)
        treated = np.asarray(treated).reshape(-1)
        post = np.asarray(post).reshape(-1)
        x = [np.ones(len(y)), treated, post, treated * post]
        if controls is not None:
            controls = _as_2d(controls)
            x.extend([controls[:, i] for i in range(controls.shape[1])])
        x = np.column_stack(x)
        self.result_ = sm.OLS(y, x).fit()
        self.treatment_effect_ = float(self.result_.params[3])
        return self

    def predict(self, treated, post, controls=None):
        treated = np.asarray(treated).reshape(-1)
        post = np.asarray(post).reshape(-1)
        x = [np.ones(len(treated)), treated, post, treated * post]
        if controls is not None:
            controls = _as_2d(controls)
            x.extend([controls[:, i] for i in range(controls.shape[1])])
        x = np.column_stack(x)
        return self.result_.predict(x)


class LogitRegression:
    """Binary response model using logistic link."""

    def __init__(self):
        self.result_ = None

    def fit(self, y, exog):
        y = np.asarray(y).reshape(-1)
        x = sm.add_constant(_as_2d(exog), has_constant="add")
        self.result_ = sm.Logit(y, x).fit(disp=False)
        return self

    def predict_proba(self, exog):
        x = sm.add_constant(_as_2d(exog), has_constant="add")
        return self.result_.predict(x)

    def predict(self, exog, threshold=0.5):
        return (self.predict_proba(exog) >= threshold).astype(int)


class ProbitRegression:
    """Binary response model using probit link."""

    def __init__(self):
        self.result_ = None

    def fit(self, y, exog):
        y = np.asarray(y).reshape(-1)
        x = sm.add_constant(_as_2d(exog), has_constant="add")
        self.result_ = sm.Probit(y, x).fit(disp=False)
        return self

    def predict_proba(self, exog):
        x = sm.add_constant(_as_2d(exog), has_constant="add")
        return self.result_.predict(x)

    def predict(self, exog, threshold=0.5):
        return (self.predict_proba(exog) >= threshold).astype(int)


__all__ = [
    "IVRegression",
    "FixedEffectsRegression",
    "RandomEffectsRegression",
    "DifferenceInDifferences",
    "LogitRegression",
    "ProbitRegression",
]
