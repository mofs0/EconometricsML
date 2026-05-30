"""自然语言 → 回归代码生成器
==============================
适用场景：
  - 用自然语言描述研究设计，自动生成 Python 回归代码
  - 快速原型验证，避免手写模板代码
  - 教学演示：展示如何将研究问题映射到计量方法

实现方式：
  1. 规则匹配（无需 API KEY，离线可用）：根据关键词识别方法
  2. API 模式（需要 OPENAI_API_KEY）：调用 GPT/Claude 生成代码

关键词识别逻辑：
  - "双重差分" / "did" / "政策效应"     → DiD
  - "工具变量" / "iv" / "内生性"        → IV/2SLS
  - "断点回归" / "rd" / "临界值"        → RD
  - "倾向得分" / "psm" / "匹配"        → PSM
  - "面板" / "固定效应" / "fe"         → Panel FE
  - "事件研究" / "累计超额收益"         → Event Study
  - 默认                              → OLS
"""
from __future__ import annotations

import re
from typing import Literal, Optional


# 方法 → 代码模板映射
_TEMPLATES = {
    "ols": '''\
from empirlab.traditional.ols import OLS
import pandas as pd

# 读取数据
df = pd.read_csv("your_data.csv")

# 定义变量
y = df["{y_var}"]
X = df[[{x_vars}]]

# 估计 OLS
model = OLS(robust=True).fit(X, y)
print(model.summary_table().round(4))
''',
    "did": '''\
from empirlab.traditional.did import DiD
import pandas as pd

# 读取面板数据（需含 unit_id, time, treated, post 列）
df = pd.read_csv("your_panel_data.csv")

model = DiD(
    outcome="{y_var}",
    treatment="treated",
    post="post",
    controls=[{x_vars}],
    entity="unit_id",
    time="time",
)
result = model.fit(df)
print(result.summary_table().round(4))
''',
    "iv": '''\
from empirlab.traditional.iv import IV2SLS
import pandas as pd

df = pd.read_csv("your_data.csv")

model = IV2SLS(
    endog="{y_var}",
    exog=[{x_vars}],
    instruments=[{z_vars}],
    robust=True,
)
result = model.fit(df)
print(result.summary_table().round(4))
''',
    "rd": '''\
from empirlab.traditional.rd import RDD
import pandas as pd

df = pd.read_csv("your_data.csv")

model = RDD(
    outcome="{y_var}",
    running="running_var",
    cutoff=0.0,
    bandwidth=None,  # None = IK 最优带宽
)
result = model.fit(df)
print(result.summary())
''',
    "psm": '''\
from empirlab.traditional.psm import PSM
import pandas as pd

df = pd.read_csv("your_data.csv")

model = PSM(
    outcome="{y_var}",
    treatment="treatment",
    covariates=[{x_vars}],
)
result = model.fit(df)
print(result.summary())
''',
    "panel": '''\
from empirlab.traditional.panel import PanelFE
import pandas as pd

df = pd.read_csv("your_panel_data.csv")

model = PanelFE(
    outcome="{y_var}",
    regressors=[{x_vars}],
    entity="firm_id",
    time="year",
    effects="two-way",
)
result = model.fit(df)
print(result.summary_table().round(4))
''',
    "event_study": '''\
from empirlab.traditional.event_study import EventStudy
import pandas as pd

df = pd.read_csv("your_event_data.csv")

model = EventStudy(
    returns="ret",
    market_returns="mkt_ret",
    event_date="event_date",
    window=(-10, 10),
)
result = model.fit(df)
result.plot_car()
''',
    "dml": '''\
from empirlab.ml.double_ml import DoubleML
import pandas as pd

df = pd.read_csv("your_data.csv")

model = DoubleML(n_folds=5)
result = model.fit(
    X=df[[{x_vars}]],
    D=df["{d_var}"],
    Y=df["{y_var}"],
)
print(result.summary().to_string(index=False))
''',
}

# 关键词 → 方法映射
_KEYWORDS = {
    "did":          ["双重差分", "did", "difference-in-difference", "政策效应", "处理组", "对照组"],
    "iv":           ["工具变量", "2sls", "iv", "内生性", "两阶段"],
    "rd":           ["断点回归", "rdd", "rd", "临界值", "regression discontinuity"],
    "psm":          ["倾向得分", "psm", "propensity score", "匹配", "matching"],
    "panel":        ["面板", "固定效应", "fe", "panel", "个体效应", "时间效应"],
    "event_study":  ["事件研究", "event study", "累计超额收益", "car", "abnormal return"],
    "dml":          ["double ml", "dml", "双重机器学习", "debiased"],
}


class TextToRegression:
    """自然语言描述 → 回归方法 + 代码生成器。

    参数
    ----
    mode : "rule" or "api", default "rule"
        "rule"：离线规则匹配（无需 API KEY）
        "api" ：调用 LLM API（需要设置 OPENAI_API_KEY 环境变量）

    示例
    ----
    >>> from empirlab.llm import TextToRegression
    >>> gen = TextToRegression()
    >>> result = gen.generate(
    ...     description="我想研究最低工资政策对就业的影响，用双重差分法",
    ...     y_var="employment",
    ...     x_vars=["log_wage", "age", "education"],
    ... )
    >>> print(result["method"])
    >>> print(result["code"])
    """

    def __init__(self, mode: Literal["rule", "api"] = "rule"):
        self.mode = mode

    def _detect_method(self, text: str) -> str:
        """规则匹配识别计量方法。"""
        text_lower = text.lower()
        for method, keywords in _KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return method
        return "ols"

    def generate(
        self,
        description: str,
        y_var: str = "y",
        x_vars: Optional[list] = None,
        d_var: str = "treatment",
        z_vars: Optional[list] = None,
    ) -> dict:
        """生成回归代码。

        参数
        ----
        description : 研究描述（自然语言）
        y_var       : 结果变量名
        x_vars      : 控制变量列表
        d_var       : 处理变量名（IV/DML 使用）
        z_vars      : 工具变量列表（IV 使用）

        返回
        ----
        dict: {method, code, description}
        """
        x_vars = x_vars or ["x1", "x2", "x3"]
        z_vars = z_vars or ["z1"]

        x_str = ", ".join(f'"{v}"' for v in x_vars)
        z_str = ", ".join(f'"{v}"' for v in z_vars)

        if self.mode == "rule":
            method = self._detect_method(description)
        else:
            method = self._api_detect(description)

        template = _TEMPLATES.get(method, _TEMPLATES["ols"])
        code = template.format(
            y_var=y_var,
            x_vars=x_str,
            d_var=d_var,
            z_vars=z_str,
        )

        method_names = {
            "ols": "普通最小二乘 (OLS)",
            "did": "双重差分 (DiD)",
            "iv": "工具变量 / 2SLS",
            "rd": "断点回归 (RDD)",
            "psm": "倾向得分匹配 (PSM)",
            "panel": "面板固定效应",
            "event_study": "事件研究法",
            "dml": "双重机器学习 (DML)",
        }

        return {
            "method": method_names.get(method, method),
            "method_key": method,
            "code": code,
            "description": description,
        }

    def _api_detect(self, description: str) -> str:
        """调用 OpenAI API 检测方法（需要 OPENAI_API_KEY）。"""
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("未找到 OPENAI_API_KEY，回退到规则匹配。")
            return self._detect_method(description)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                "你是计量经济学专家。根据以下研究描述，判断最适合的计量方法，"
                f"从以下选项中选一个：{list(_TEMPLATES.keys())}。"
                f"只输出方法名，不要其他内容。\n\n描述：{description}"
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )
            method = resp.choices[0].message.content.strip().lower()
            return method if method in _TEMPLATES else self._detect_method(description)
        except Exception as e:
            print(f"API 调用失败（{e}），回退到规则匹配。")
            return self._detect_method(description)


if __name__ == "__main__":
    gen = TextToRegression(mode="rule")

    cases = [
        ("研究最低工资政策对就业的影响，用双重差分法", "employment", ["log_mw", "gdp", "unemp"]),
        ("用工具变量估计教育对收入的因果效应", "income", ["educ", "exper"], "educ", ["distance"]),
        ("分析企业面板数据，用固定效应控制个体异质性", "tfp", ["rd", "size", "lev"]),
        ("我想做一个简单的OLS回归", "y", ["x1", "x2"]),
    ]

    for case in cases:
        desc = case[0]
        y = case[1]
        x = case[2]
        d = case[3] if len(case) > 3 else "treatment"
        z = case[4] if len(case) > 4 else ["z1"]
        result = gen.generate(desc, y_var=y, x_vars=x, d_var=d, z_vars=z)
        print(f"\n描述：{desc}")
        print(f"识别方法：{result['method']}")
        print("生成代码：")
        print(result["code"])
        print("-" * 60)
