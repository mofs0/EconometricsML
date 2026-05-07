"""LLM 文本分类调用模板（不训练，仅示例如何封装调用）。

注：实际调用外部 LLM（如 OpenAI）时需在环境中配置 API key。
"""
from typing import List


class TextClassifier:
    """封装式接口示例：接受文本列表，返回分类标签（占位）。"""

    def __init__(self, model_name: str = "gpt-like"):
        self.model_name = model_name

    def classify(self, texts: List[str]) -> List[str]:
        """占位实现：直接返回空标签列表，真实实现应调用 LLM API 并解析结果。"""
        return ["unknown" for _ in texts]


if __name__ == "__main__":
    tc = TextClassifier()
    sample = ["公司营收增长 20%", "宏观环境恶化导致下行风险"]
    print(tc.classify(sample))
