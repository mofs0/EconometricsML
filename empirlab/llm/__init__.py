"""empirlab.llm —— 大语言模型辅助计量研究子包。

模块列表
--------
text_to_regression : 自然语言描述 → 自动生成回归代码
sentiment_index    : 文本情感指数构建（用于计量分析的文本变量）
"""

from .text_to_regression import TextToRegression
from .sentiment_index import SentimentIndex

__all__ = ["TextToRegression", "SentimentIndex"]
