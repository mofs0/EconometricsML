"""
提示词调用客户端 (Prompt Client)
=================================
来源参考：OpenAI API / LLM Prompting Best Practices
适用场景：统一封装 LLM 调用参数与响应结构
Python 版本：3.10+
依赖：numpy >= 1.21 / pandas >= 1.3
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from empirlab.utils.metrics import calculate_metrics


class PromptClient:
    """最小 LLM 调用占位实现。"""

    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
        self.is_fitted = False
        self._history: list[dict[str, str]] = []

    def fit(self, prompts, **kwargs):
        """保存提示词历史作为调用基线。"""
        prompt_list = [str(p) for p in prompts]
        self._history = [{"role": "user", "content": p} for p in prompt_list]
        self.is_fitted = True
        return self

    def predict(self, prompts):
        """返回占位响应，便于离线流程联调。"""
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 方法。")
        prompt_list = [str(p) for p in prompts]
        return [f"[{self.model_name}] placeholder response: {p[:60]}" for p in prompt_list]

    def summary(self):
        """返回调用摘要。"""
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 方法。")
        lengths = np.array([len(item["content"]) for item in self._history], dtype=float)
        baseline = np.full_like(lengths, lengths.mean()) if lengths.size else np.array([0.0])
        metrics = calculate_metrics(lengths if lengths.size else np.array([0.0]), baseline)
        return {
            "model": "PromptClient",
            "backend": self.model_name,
            "n_prompts": len(self._history),
            "prompt_length_stats": metrics,
        }


if __name__ == "__main__":
    model = PromptClient()
    prompts = [
        "请根据给定回归结果写出一段学术风格结论。",
        "请给出稳健性检验建议。",
    ]
    model.fit(prompts)
    result = model.summary()
    print("模型摘要:", pd.Series(result).to_dict())
    print("预测示例:", model.predict(["解释 OLS 系数的经济含义"])[0])
