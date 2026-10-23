"""
轻量 BM25 稀疏检索（不新增第三方依赖）。

为什么需要稀疏路（v2 §平面二）：
- 纯向量检索会把订单号、型号、条款版本号、专有名词“平滑”掉；
- BM25 对词面匹配极其敏感，是精确标识符查询的兜底。

中文分词策略：
- 不使用 jieba（避免新依赖与词典漂移）；
- 对连续汉字同时生成 **unigram + bigram**：既保留单字召回，又能让「为政」「学而」这类
  双字词获得更高权重；英文/数字/型号按连续 token 处理，保留 `HD-8842`、`v2.1` 等标识符。
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict

from langchain_core.documents import Document

_ASCII_TOKEN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:/#\-]*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens: list[str] = [m.group(0) for m in _ASCII_TOKEN.finditer(lowered)]
    for run in _CJK_RUN.findall(lowered):
        chars = list(run)
        tokens.extend(chars)
        tokens.extend("".join(pair) for pair in zip(chars, chars[1:]))
    return tokens


class BM25Index:
    """Okapi BM25，支持按 allowed_ids 做召回前过滤（ACL 语义）。"""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.chunk_ids = [str(doc.metadata.get("chunk_id", "")) for doc in documents]
        self.doc_tokens = [tokenize(doc.page_content) for doc in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.n_docs = len(documents)
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        df: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, tf in enumerate(self.term_freqs):
            for term, freq in tf.items():
                df[term] += 1
                self.postings[term].append((idx, freq))

        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(
        self,
        query: str,
        k: int = 10,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[Document, float]]:
        if not self.n_docs:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, freq in self.postings[term]:
                if allowed_ids is not None and self.chunk_ids[idx] not in allowed_ids:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / (self.avgdl or 1))
                scores[idx] += idf * freq * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self.documents[idx], score) for idx, score in ranked]


_bm25_cache: dict[str, BM25Index] = {}
_bm25_lock = threading.Lock()


def get_bm25_index(cache_key: str, documents: list[Document]) -> BM25Index:
    """按 cache_key（索引版本 + 块数）缓存 BM25，避免每次查询重建。"""
    with _bm25_lock:
        cached = _bm25_cache.get(cache_key)
        if cached is not None:
            return cached
    index = BM25Index(documents)
    with _bm25_lock:
        _bm25_cache[cache_key] = index
    return index


def clear_bm25_cache() -> None:
    with _bm25_lock:
        _bm25_cache.clear()
