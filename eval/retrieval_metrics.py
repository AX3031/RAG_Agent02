"""
检索侧指标（纯标准库实现，离线、确定、可复现）。

指标定义沿用信息检索领域的通用口径（TREC / BEIR）：
- Recall@K：Top-K 命中的相关块数 / 全部相关块数；
- HitRate@K：Top-K 是否至少命中一个相关块；
- MRR@K：第一个相关块排名的倒数；
- NDCG@K：二元相关性下的归一化折损累计增益。

标注以「块」为单位，但 Golden Set 存的是**证据片段**：
评测时把片段归一化（去空白/标点）后映射到 chunk_id，
这样切分参数变化时标注仍可复用（v2 §4.5 的可迁移性考虑）。
"""

from __future__ import annotations

import math
import re

_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """去掉空白与标点，保留中英文与数字；用于片段与块的模糊定位。"""
    return _NORMALIZE_RE.sub("", (text or "").lower())


def resolve_snippets(
    snippets: list[str],
    documents: list,
    max_matches_per_snippet: int = 5,
    min_chunk_chars: int = 0,
) -> tuple[set[str], list[str]]:
    """
    把证据片段解析成 chunk_id 集合。

    返回 (chunk_ids, unresolved_snippets)。
    - 片段能直接在某块中找到：取包含它的块（过多时保留最短的若干块，避免短词过匹配）；
    - 找不到（可能跨块）：按标点切成子片段后递归解析。
    """
    resolved: set[str] = set()
    unresolved: list[str] = []
    normalized_docs = [
        (str(doc.metadata.get("chunk_id") or ""), normalize(doc.page_content), len(doc.page_content))
        for doc in documents
    ]

    def resolve_one(snippet: str, depth: int = 0) -> bool:
        target = normalize(snippet)
        if not target:
            return False
        # 单个汉字会过度匹配，但单个英文/希腊字符（如 κ、v）是有效标识符
        if len(target) < 2 and re.search(r"[\u4e00-\u9fff]", snippet):
            return False
        matches = [
            (cid, length)
            for cid, text, length in normalized_docs
            if cid and target in text and length >= min_chunk_chars
        ]
        if not matches and min_chunk_chars > 0:
            # 语料里可能只有短块包含该片段，退化为不过滤长度
            matches = [(cid, length) for cid, text, length in normalized_docs if cid and target in text]
        if matches:
            if len(matches) > max_matches_per_snippet:
                matches = sorted(matches, key=lambda item: item[1])[:max_matches_per_snippet]
            resolved.update(cid for cid, _ in matches)
            return True
        if depth >= 2:
            return False
        parts = [part for part in re.split(r"[，。！？；：、\s]+", snippet) if len(normalize(part)) >= 4]
        if not parts:
            return False
        found = False
        for part in parts:
            found = resolve_one(part, depth + 1) or found
        return found

    for snippet in snippets:
        if not resolve_one(snippet):
            unresolved.append(snippet)
    return resolved, unresolved


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def random_recall_at_k(total_chunks: int, relevant_count: int, k: int) -> float:
    """随机检索的期望 Recall@K，用来判断小语料上的高分是否只是“基数效应”。"""
    if total_chunks <= 0 or relevant_count <= 0:
        return 0.0
    if k >= total_chunks:
        return 1.0
    # 1 - C(N-R, k) / C(N, k)
    try:
        miss = math.comb(total_chunks - relevant_count, k) / math.comb(total_chunks, k)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return 1.0 - miss


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]
