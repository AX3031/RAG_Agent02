"""
检索链路的可观测性与短期缓存。

可观测（v2 §8）：
- 每次检索生成 request_id / trace_id，贯穿各阶段；
- 记录 rewrite/dense/sparse/fuse/rerank/expand/assemble 的分段耗时；
- 日志只写 query_hash、doc_id、evidence_id、状态与延迟，**不写完整 query 与证据正文**。

缓存（v2 §7）：
- 进程内 TTL + LRU；
- key 必须包含 query、scope、index_version、top_k 与检索参数，避免文档更新后命中脏证据、也避免越权命中。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from utils.logger_handler import logger


def hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


class Trace:
    def __init__(self, name: str = "rag.retrieval"):
        self.name = name
        self.request_id = f"req_{uuid4().hex[:12]}"
        self.trace_id = f"trace_{uuid4().hex[:12]}"
        self.started_at = time.perf_counter()
        self.stages: dict[str, float] = {}
        self.degraded: list[str] = []

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - start) * 1000, 2)

    def mark_degraded(self, stage: str) -> None:
        if stage not in self.degraded:
            self.degraded.append(stage)

    def latency(self) -> dict[str, float]:
        total = round((time.perf_counter() - self.started_at) * 1000, 2)
        result = dict(self.stages)
        result["total"] = total
        return result

    def log(self, *, query: str, status: str, answerable: bool, evidence: list) -> None:
        payload = {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "query_hash": hash_text(query),
            "query_chars": len(query or ""),
            "status": status,
            "answerable": bool(answerable),
            "evidence_ids": [item.evidence_id for item in evidence],
            "doc_ids": sorted({item.doc_id for item in evidence}),
            "degraded_stages": self.degraded,
            "latency_ms": self.latency(),
        }
        logger.info("[RAG_TRACE] %s", json.dumps(payload, ensure_ascii=False))


class TTLCache:
    """极简 TTL + LRU 缓存；单进程可用，多实例部署时应换 Redis（v2 §7）。"""

    def __init__(self, max_size: int = 128, ttl_seconds: int = 300):
        self.max_size = max(1, max_size)
        self.ttl_seconds = max(0.001, ttl_seconds)
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def _now(self) -> float:
        return time.monotonic()

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < self._now():
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (self._now() + self.ttl_seconds, value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


def make_cache_key(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
