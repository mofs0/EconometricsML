"""文本情感指数构建。

方法：LM 金融词典（离线）/ SnowNLP / BERT
"""
from __future__ import annotations
from typing import List, Literal
import numpy as np
import pandas as pd

_LM_POSITIVE = {
    "achieve","advantage","benefit","best","better","booming","confidence",
    "effective","efficient","excellent","exceed","favorable","gain","good",
    "great","growth","improve","increase","leading","optimal","outperform",
    "outstanding","positive","profitable","progress","promising","recover",
    "robust","stable","strong","success","superior","sustainable",
    "上升","增长","盈利","优秀","稳定","改善","突破","领先",
    "创新","优势","积极","显著","强劲","超额","提升","卓越",
}

_LM_NEGATIVE = {
    "adverse","below","concern","decline","decrease","difficult","doubt",
    "fail","falling","harm","impair","insufficient","lack","limit","loss",
    "lower","negative","obstacle","penalty","poor","problem","reduce",
    "risk","shortage","slow","uncertain","underperform","unfavorable",
    "unstable","vulnerable","weak","worse","writeoff",
    "下降","亏损","风险","困难","不确定","减少","问题","损失",
    "警告","负面","违约","下滑","萎缩","压力","不足","滞后",
}


class SentimentIndex:
    """文本情感指数构建器。

    参数
    ----
    method : 'lm' | 'snownlp' | 'bert'
    lang   : 'zh' | 'en' | 'auto'

    示例
    ----
    >>> si = SentimentIndex(method='lm')
    >>> si.score(['公司业绩增长盈利提升', '亏损下滑风险压力'])
    """

    def __init__(self, method: Literal["lm","snownlp","bert"] = "lm",
                 lang: Literal["zh","en","auto"] = "auto"):
        self.method = method
        self.lang = lang
        self._bert_model = None
        self._bert_tokenizer = None

    def score(self, texts: List[str]) -> np.ndarray:
        """返回情感分数数组，正=正面，负=负面。"""
        if self.method == "lm":
            return np.array([self._lm_score(t) for t in texts])
        elif self.method == "snownlp":
            return self._snownlp_scores(texts)
        elif self.method == "bert":
            return self._bert_scores(texts)
        raise ValueError(f"未知 method: {self.method}")

    def score_df(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """在 DataFrame 上添加 sentiment_score / sentiment_label 列。"""
        scores = self.score(df[text_col].tolist())
        result = df.copy()
        result["sentiment_score"] = scores
        result["sentiment_label"] = pd.cut(
            scores, bins=[-np.inf, -0.1, 0.1, np.inf],
            labels=["negative","neutral","positive"],
        )
        return result

    def _lm_score(self, text: str) -> float:
        """子串匹配 LM 词典法，归一化到文本字符数/4。"""
        t = text.lower()
        try:
            import jieba
            tokens = list(jieba.cut(t))
            pos = sum(1 for w in tokens if w in _LM_POSITIVE)
            neg = sum(1 for w in tokens if w in _LM_NEGATIVE)
            return (pos - neg) / max(len(tokens), 1)
        except ImportError:
            pass
        pos = sum(1 for w in _LM_POSITIVE if w in t)
        neg = sum(1 for w in _LM_NEGATIVE if w in t)
        return (pos - neg) / max(len(t) // 4, 1)

    def _snownlp_scores(self, texts: List[str]) -> np.ndarray:
        try:
            from snownlp import SnowNLP
        except ImportError:
            raise ImportError("pip install snownlp")
        return 2 * np.array([SnowNLP(t).sentiments for t in texts]) - 1

    def _bert_scores(self, texts: List[str]) -> np.ndarray:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("pip install transformers torch")
        if self._bert_model is None:
            nm = "bert-base-chinese" if self.lang in ("zh","auto") else "bert-base-uncased"
            self._bert_tokenizer = AutoTokenizer.from_pretrained(nm)
            self._bert_model = AutoModel.from_pretrained(nm)
            self._bert_model.eval()
        pos_a = "公司业绩优秀盈利增长" if self.lang in ("zh","auto") else "excellent performance strong growth"
        neg_a = "公司风险损失业绩下滑" if self.lang in ("zh","auto") else "poor performance significant losses"

        def emb(s):
            inp = self._bert_tokenizer(s, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = self._bert_model(**inp)
            return out.last_hidden_state[:,0,:].squeeze().numpy()

        vp, vn = emb(pos_a), emb(neg_a)
        out = []
        for t in texts:
            v = emb(t)
            cp = float(np.dot(v, vp) / (np.linalg.norm(v) * np.linalg.norm(vp) + 1e-8))
            cn = float(np.dot(v, vn) / (np.linalg.norm(v) * np.linalg.norm(vn) + 1e-8))
            out.append(cp - cn)
        return np.array(out)


if __name__ == "__main__":
    si = SentimentIndex(method="lm")
    texts = ["公司业绩持续增长，盈利能力显著提升", "面临较大风险不确定性，业绩下滑亏损压力", "公司经营稳定"]
    for t, s in zip(texts, si.score(texts)):
        label = "正面" if s > 0.05 else ("负面" if s < -0.05 else "中性")
        print(f"[{label}] {s:+.4f}  {t}")
