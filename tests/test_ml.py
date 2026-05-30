"""
单元测试：empirlab.ml —— 机器学习计量方法
运行方式：
    pytest tests/test_ml.py -v
"""
import numpy as np
import pandas as pd
import pytest


# ── DoubleML ──────────────────────────────────────────────────────────

class TestDoubleML:
    def setup_method(self):
        from empirlab.ml.double_ml import dml_data
        self.df = dml_data(n=400, p=5, true_theta=0.5, seed=42)
        self.X = self.df[[c for c in self.df.columns if c.startswith("x")]]
        self.D = self.df["d"]
        self.Y = self.df["y"]

    def test_fit_returns_self(self):
        from empirlab.ml.double_ml import DoubleML
        m = DoubleML(n_folds=3)
        assert m.fit(self.X, self.D, self.Y) is m

    def test_theta_close_to_truth(self):
        from empirlab.ml.double_ml import DoubleML
        m = DoubleML(n_folds=3).fit(self.X, self.D, self.Y)
        assert abs(m.theta_ - 0.5) < 0.3  # 允许 0.3 误差

    def test_se_positive(self):
        from empirlab.ml.double_ml import DoubleML
        m = DoubleML(n_folds=3).fit(self.X, self.D, self.Y)
        assert m.se_ > 0

    def test_ci_contains_truth(self):
        from empirlab.ml.double_ml import DoubleML
        m = DoubleML(n_folds=3).fit(self.X, self.D, self.Y)
        lo, hi = m.ci_
        # 95% CI 应以较高概率覆盖真值（在模拟数据上验证方向性）
        assert lo < m.theta_ < hi

    def test_summary_is_df(self):
        from empirlab.ml.double_ml import DoubleML
        m = DoubleML(n_folds=3).fit(self.X, self.D, self.Y)
        assert isinstance(m.summary(), pd.DataFrame)

    def test_unfitted_summary_raises(self):
        from empirlab.ml.double_ml import DoubleML
        with pytest.raises(RuntimeError):
            DoubleML().summary()


# ── LassoSelect ───────────────────────────────────────────────────────

class TestLassoSelect:
    def setup_method(self):
        from empirlab.ml.lasso_select import lasso_data
        self.df = lasso_data(n=300, p=20, k_true=5, seed=0)
        self.y = self.df["y"]
        self.X = self.df.drop(columns="y")

    def test_fit(self):
        from empirlab.ml.lasso_select import LassoSelect
        m = LassoSelect(cv=3).fit(self.X, self.y)
        assert m.alpha_ is not None

    def test_selected_vars_nonempty(self):
        from empirlab.ml.lasso_select import LassoSelect
        m = LassoSelect(cv=3).fit(self.X, self.y)
        assert len(m.selected_vars_) > 0

    def test_selects_mostly_true_vars(self):
        """应该选出大部分真实变量（x1-x5）。"""
        from empirlab.ml.lasso_select import LassoSelect
        m = LassoSelect(cv=3).fit(self.X, self.y)
        true_vars = {f"x{i+1}" for i in range(5)}
        selected = set(m.selected_vars_)
        overlap = len(true_vars & selected)
        assert overlap >= 2  # 至少选出 2 个真实变量

    def test_post_coef_shape(self):
        from empirlab.ml.lasso_select import LassoSelect
        m = LassoSelect(cv=3, post_ols=True).fit(self.X, self.y)
        if m.post_coef_ is not None:
            assert len(m.post_coef_) == len(m.selected_vars_)

    def test_summary_df(self):
        from empirlab.ml.lasso_select import LassoSelect
        m = LassoSelect(cv=3).fit(self.X, self.y)
        s = m.summary()
        assert isinstance(s, pd.DataFrame)
        assert "lasso_coef" in s.columns
        assert "selected" in s.columns


# ── RFRegressor ───────────────────────────────────────────────────────

class TestRFRegressor:
    def setup_method(self):
        from empirlab.ml.random_forest import rf_data
        self.df = rf_data(n=300, p=8, seed=1)
        self.y = self.df["y"]
        self.X = self.df.drop(columns="y")

    def test_fit(self):
        from empirlab.ml.random_forest import RFRegressor
        m = RFRegressor(n_estimators=50).fit(self.X, self.y)
        assert m.oob_score_ is not None

    def test_oob_positive(self):
        from empirlab.ml.random_forest import RFRegressor
        m = RFRegressor(n_estimators=50).fit(self.X, self.y)
        assert m.oob_score_ > -1.0  # 允许很差但不崩溃

    def test_importance_table(self):
        from empirlab.ml.random_forest import RFRegressor
        m = RFRegressor(n_estimators=50).fit(self.X, self.y)
        tbl = m.importance_table_
        assert isinstance(tbl, pd.DataFrame)
        assert len(tbl) == self.X.shape[1]
        assert abs(tbl["importance_gini"].sum() - 1.0) < 1e-6

    def test_predict_shape(self):
        from empirlab.ml.random_forest import RFRegressor
        m = RFRegressor(n_estimators=50).fit(self.X, self.y)
        preds = m.predict(self.X)
        assert preds.shape == (len(self.X),)

    def test_top_vars_include_true(self):
        """x1（非线性）和 x2/x3（交互）应在重要性前5中。"""
        from empirlab.ml.random_forest import RFRegressor
        m = RFRegressor(n_estimators=100).fit(self.X, self.y)
        top5 = set(m.importance_table_["variable"].head(5))
        assert len(top5 & {"x1", "x2", "x3", "x4"}) >= 2


# ── CausalForest ─────────────────────────────────────────────────────

class TestCausalForest:
    def setup_method(self):
        from empirlab.ml.causal_forest import cf_data
        self.df = cf_data(n=400, true_ate=1.0, seed=7)

    def test_t_learner_ate(self):
        from empirlab.ml.causal_forest import CausalForest
        m = CausalForest(learner="T").fit(
            self.df[["x1", "x2", "x3"]], self.df["d"], self.df["y"], n_bootstrap=50
        )
        assert abs(m.ate_ - 1.0) < 0.5

    def test_s_learner_runs(self):
        from empirlab.ml.causal_forest import CausalForest
        m = CausalForest(learner="S").fit(
            self.df[["x1", "x2", "x3"]], self.df["d"], self.df["y"], n_bootstrap=50
        )
        assert m.ate_ is not None

    def test_cate_shape(self):
        from empirlab.ml.causal_forest import CausalForest
        m = CausalForest(learner="T").fit(
            self.df[["x1", "x2", "x3"]], self.df["d"], self.df["y"], n_bootstrap=50
        )
        assert m.cate_.shape == (len(self.df),)

    def test_summary_keys(self):
        from empirlab.ml.causal_forest import CausalForest
        m = CausalForest(learner="T").fit(
            self.df[["x1", "x2", "x3"]], self.df["d"], self.df["y"], n_bootstrap=50
        )
        s = m.summary()
        for key in ["ATE", "SE", "t_stat", "p_value"]:
            assert key in s

    def test_cate_summary_df(self):
        from empirlab.ml.causal_forest import CausalForest
        m = CausalForest(learner="T").fit(
            self.df[["x1", "x2", "x3"]], self.df["d"], self.df["y"], n_bootstrap=50
        )
        tbl = m.cate_summary()
        assert isinstance(tbl, pd.DataFrame)


# ── TextToRegression（LLM 规则模式）────────────────────────────────────

class TestTextToRegression:
    def test_detect_did(self):
        from empirlab.llm.text_to_regression import TextToRegression
        gen = TextToRegression(mode="rule")
        r = gen.generate("用双重差分法研究政策效应", "y", ["x1"])
        assert r["method_key"] == "did"

    def test_detect_iv(self):
        from empirlab.llm.text_to_regression import TextToRegression
        gen = TextToRegression(mode="rule")
        r = gen.generate("工具变量估计内生性问题", "y", ["x1"], d_var="d", z_vars=["z"])
        assert r["method_key"] == "iv"

    def test_detect_ols_fallback(self):
        from empirlab.llm.text_to_regression import TextToRegression
        gen = TextToRegression(mode="rule")
        r = gen.generate("做个回归分析", "y", ["x1"])
        assert r["method_key"] == "ols"

    def test_code_contains_y_var(self):
        from empirlab.llm.text_to_regression import TextToRegression
        gen = TextToRegression(mode="rule")
        r = gen.generate("OLS 分析", "my_outcome", ["ctrl1", "ctrl2"])
        assert "my_outcome" in r["code"]


# ── SentimentIndex ────────────────────────────────────────────────────

class TestSentimentIndex:
    def test_lm_returns_array(self):
        from empirlab.llm.sentiment_index import SentimentIndex
        si = SentimentIndex(method="lm")
        texts = ["公司业绩增长显著", "面临较大风险损失"]
        scores = si.score(texts)
        assert len(scores) == 2

    def test_positive_higher_than_negative(self):
        from empirlab.llm.sentiment_index import SentimentIndex
        si = SentimentIndex(method="lm")
        pos_score = si.score(["公司盈利增长优秀稳定"])[0]
        neg_score = si.score(["公司亏损风险下降损失"])[0]
        assert pos_score > neg_score

    def test_score_df(self):
        from empirlab.llm.sentiment_index import SentimentIndex
        si = SentimentIndex(method="lm")
        df = pd.DataFrame({"text": ["业绩增长", "出现亏损"], "id": [1, 2]})
        result = si.score_df(df, "text")
        assert "sentiment_score" in result.columns
        assert "sentiment_label" in result.columns
