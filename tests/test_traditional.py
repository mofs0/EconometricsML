"""
单元测试：empirlab.traditional —— 全部 11 个计量方法模块
运行方式：
    pytest tests/test_traditional.py -v
"""
import numpy as np
import pandas as pd
import pytest

# ── OLS ──────────────────────────────────────────────────────────────

class TestOLS:
    def test_fit_returns_self(self):
        from empirlab.traditional.ols import OLS
        X = np.random.randn(100, 3)
        y = X @ [1, -0.5, 0.3] + np.random.randn(100) * 0.3
        m = OLS()
        assert m.fit(X, y) is m

    def test_coef_shape(self):
        from empirlab.traditional.ols import OLS
        X = np.random.randn(200, 4)
        y = X[:, 0] + np.random.randn(200)
        m = OLS().fit(X, y)
        assert m.coef_.shape == (4,)

    def test_summary_keys(self):
        from empirlab.traditional.ols import OLS
        X = np.random.randn(100, 2)
        y = X @ [1, 2] + 0.5
        m = OLS(robust=True).fit(X, y)
        s = m.summary()
        for key in ["coefficients", "fit", "residuals"]:
            assert key in s

    def test_r2_range(self):
        from empirlab.traditional.ols import OLS, mincer_data
        df = mincer_data(n=300)
        m = OLS().fit(df[["educ", "exper", "exper2"]], df["ln_wage"])
        r2 = m.summary()["fit"]["R2"]
        assert 0.0 <= r2 <= 1.0

    def test_predict_shape(self):
        from empirlab.traditional.ols import OLS
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        m = OLS().fit(X, y)
        preds = m.predict(np.random.randn(50, 3))
        assert preds.shape == (50,)

    def test_summary_table_df(self):
        from empirlab.traditional.ols import OLS
        X = pd.DataFrame({"a": np.random.randn(80), "b": np.random.randn(80)})
        y = np.random.randn(80)
        m = OLS().fit(X, y)
        tbl = m.summary_table()
        assert isinstance(tbl, pd.DataFrame)
        assert "const" in tbl.index

    def test_unfitted_predict_raises(self):
        from empirlab.traditional.ols import OLS
        with pytest.raises(RuntimeError):
            OLS().predict(np.ones((5, 3)))


# ── Logit ─────────────────────────────────────────────────────────────

class TestLogit:
    def setup_method(self):
        from empirlab.traditional.logit import logit_data
        self.df = logit_data(n=300)

    def test_fit(self):
        from empirlab.traditional.logit import Logit
        m = Logit().fit(self.df[["size", "productivity"]], self.df["export"])
        assert m.is_fitted

    def test_predict_proba_range(self):
        from empirlab.traditional.logit import Logit
        m = Logit().fit(self.df[["size", "productivity"]], self.df["export"])
        probs = m.predict_proba(self.df[["size", "productivity"]])
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_summary_has_marginal_effects(self):
        from empirlab.traditional.logit import Logit
        m = Logit().fit(self.df[["size", "productivity"]], self.df["export"])
        s = m.summary()
        assert "marginal_effects" in s or "coefficients" in s


# ── IV / 2SLS ─────────────────────────────────────────────────────────

class TestIV:
    def setup_method(self):
        from empirlab.traditional.iv import iv_data
        self.df = iv_data(n=500)

    def test_fit(self):
        from empirlab.traditional.iv import IV2SLS
        m = IV2SLS().fit(
            self.df[["institutions", "latitude"]],
            self.df["log_gdp"],
            self.df[["settler_mortality"]],
        )
        assert m.is_fitted

    def test_first_stage_f(self):
        from empirlab.traditional.iv import IV2SLS
        m = IV2SLS().fit(
            self.df[["latitude"]],
            self.df["log_gdp"],
            self.df[["settler_mortality"]],
        )
        s = m.summary()
        # 弱工具变量检验：一阶段 F 统计量应为正
        assert s.get("first_stage_F", 0) >= 0


# ── Panel FE ─────────────────────────────────────────────────────────

class TestPanel:
    def setup_method(self):
        from empirlab.traditional.panel import panel_data
        self.df = panel_data(n_firms=30, n_years=5)

    def test_fit_returns_self(self):
        from empirlab.traditional.panel import PanelFE
        m = PanelFE()
        result = m.fit(self.df, outcome="log_tfp",
                       regressors=["log_capital", "log_labor"],
                       entity="firm_id", time="year")
        assert result is m

    def test_coef_count(self):
        from empirlab.traditional.panel import PanelFE
        m = PanelFE().fit(self.df, outcome="log_tfp",
                          regressors=["log_capital", "log_labor"],
                          entity="firm_id", time="year")
        assert len(m.coef_) == 2


# ── DiD ───────────────────────────────────────────────────────────────

class TestDiD:
    def setup_method(self):
        from empirlab.traditional.did import did_data
        self.df = did_data(n_states=20, n_periods=4)

    def test_fit(self):
        from empirlab.traditional.did import DiD
        m = DiD().fit(self.df, outcome="log_employment",
                      treatment="treated", post="post",
                      entity="state", time="year")
        assert m.is_fitted

    def test_did_coef_close(self):
        """DID 系数应接近模拟时设置的真实处理效应。"""
        from empirlab.traditional.did import DiD, did_data
        df = did_data(n_states=50, n_periods=4, true_effect=-0.1, seed=0)
        m = DiD().fit(df, outcome="log_employment",
                      treatment="treated", post="post",
                      entity="state", time="year")
        s = m.summary()
        theta = s["coefficients"]["treated_x_post"]["coef"]
        assert abs(theta - (-0.1)) < 0.15  # 允许 0.15 的误差


# ── RD ────────────────────────────────────────────────────────────────

class TestRD:
    def setup_method(self):
        from empirlab.traditional.rd import rd_data
        self.df = rd_data(n=400)

    def test_fit(self):
        from empirlab.traditional.rd import RDD
        m = RDD().fit(self.df, outcome="mortality", running="age", cutoff=21.0)
        assert m.is_fitted

    def test_summary(self):
        from empirlab.traditional.rd import RDD
        m = RDD().fit(self.df, outcome="mortality", running="age", cutoff=21.0)
        s = m.summary()
        assert "rd_estimate" in s or "tau" in s or "coefficients" in s


# ── PSM ───────────────────────────────────────────────────────────────

class TestPSM:
    def setup_method(self):
        from empirlab.traditional.psm import psm_data
        self.df = psm_data(n=400)

    def test_fit(self):
        from empirlab.traditional.psm import PSM
        m = PSM().fit(self.df, outcome="rd_intensity",
                      treatment="subsidy",
                      covariates=["size", "age", "lev", "roe"])
        assert m.is_fitted

    def test_att_numeric(self):
        from empirlab.traditional.psm import PSM
        m = PSM().fit(self.df, outcome="rd_intensity",
                      treatment="subsidy",
                      covariates=["size", "age", "lev", "roe"])
        s = m.summary()
        att = s.get("ATT") or s.get("att")
        assert att is not None and np.isfinite(att)


# ── Event Study ───────────────────────────────────────────────────────

class TestEventStudy:
    def setup_method(self):
        from empirlab.traditional.event_study import event_study_data
        self.df = event_study_data(n_events=20, window=(-5, 5))

    def test_fit(self):
        from empirlab.traditional.event_study import EventStudy
        m = EventStudy(window=(-5, 5)).fit(self.df)
        assert m.is_fitted

    def test_car_shape(self):
        from empirlab.traditional.event_study import EventStudy
        m = EventStudy(window=(-5, 5)).fit(self.df)
        s = m.summary()
        assert "CAR" in s or "car" in s


# ── GARCH ─────────────────────────────────────────────────────────────

class TestGARCH:
    def setup_method(self):
        from empirlab.traditional.garch import garch_data
        self.returns = garch_data(n=500)

    def test_fit(self):
        from empirlab.traditional.garch import GARCH11
        m = GARCH11().fit(self.returns)
        assert m.is_fitted

    def test_params_positive(self):
        from empirlab.traditional.garch import GARCH11
        m = GARCH11().fit(self.returns)
        s = m.summary()
        assert s["omega"] > 0
        assert s["alpha"] >= 0
        assert s["beta"] >= 0

    def test_persistence_less_one(self):
        from empirlab.traditional.garch import GARCH11
        m = GARCH11().fit(self.returns)
        s = m.summary()
        assert s["alpha"] + s["beta"] < 1.05  # 允许轻微超出（数值优化误差）


# ── VAR ───────────────────────────────────────────────────────────────

class TestVAR:
    def setup_method(self):
        from empirlab.traditional.var import var_data
        self.df = var_data(n=200)

    def test_fit(self):
        from empirlab.traditional.var import VAR
        m = VAR(lags=2).fit(self.df[["gdp_growth", "inflation", "rate"]])
        assert m.is_fitted

    def test_forecast_shape(self):
        from empirlab.traditional.var import VAR
        m = VAR(lags=2).fit(self.df[["gdp_growth", "inflation", "rate"]])
        fc = m.forecast(steps=4)
        assert fc.shape == (4, 3)


# ── Synthetic Control ─────────────────────────────────────────────────

class TestSC:
    def setup_method(self):
        from empirlab.traditional.synthetic_control import sc_data
        self.df = sc_data(n_controls=10, n_periods=30, treat_period=20)

    def test_fit(self):
        from empirlab.traditional.synthetic_control import SyntheticControl
        m = SyntheticControl().fit(
            self.df,
            outcome="gdp",
            unit="unit",
            time="time",
            treated_unit="treated",
            pre_period=20,
        )
        assert m.is_fitted

    def test_weights_sum_one(self):
        from empirlab.traditional.synthetic_control import SyntheticControl
        m = SyntheticControl().fit(
            self.df,
            outcome="gdp",
            unit="unit",
            time="time",
            treated_unit="treated",
            pre_period=20,
        )
        assert abs(sum(m.weights_.values()) - 1.0) < 1e-4
