"""
RAG 子 Agent 的接口契约（对应《对RAG的初步要求》v2 §1）。

核心变化：输出不再是“一段不可追溯的自然语言”，而是
`evidence[] + answerable + status + 可观测字段`。

状态机严格使用 v2 §1.3 的五个状态：
    ok | no_evidence | degraded | denied | error
error.code 使用 v2 §2 的失败类型：
    TIMEOUT | INDEX_STALE | UPSTREAM_5XX | RATE_LIMIT | INVALID_QUERY | DENIED | NO_EVIDENCE
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

CONTRACT_VERSION = "1.0"


class Status(str, Enum):
    OK = "ok"
    NO_EVIDENCE = "no_evidence"
    DEGRADED = "degraded"
    DENIED = "denied"
    ERROR = "error"


class ErrorCode(str, Enum):
    TIMEOUT = "TIMEOUT"
    INDEX_STALE = "INDEX_STALE"
    UPSTREAM_5XX = "UPSTREAM_5XX"
    RATE_LIMIT = "RATE_LIMIT"
    INVALID_QUERY = "INVALID_QUERY"
    DENIED = "DENIED"
    NO_EVIDENCE = "NO_EVIDENCE"


@dataclass
class Evidence:
    evidence_id: str
    doc_id: str
    version: str
    chunk_id: str
    parent_id: str
    title_path: list[str]
    page: int
    char_start: int
    char_end: int
    text: str
    context: str
    source: str
    score: dict
    scope_tags: list[str] = field(default_factory=list)
    updated_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "version": self.version,
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "title_path": self.title_path,
            "page": self.page,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
            "context": self.context,
            "source": self.source,
            "score": self.score,
            "scope_tags": self.scope_tags,
            "updated_at": self.updated_at,
        }


@dataclass
class RetrievalResult:
    status: Status
    answerable: bool
    evidence: list[Evidence]
    request_id: str
    trace_id: str
    index_version: str
    latency_ms: dict[str, float | int]
    degraded_stages: list[str] = field(default_factory=list)
    error: dict | None = None
    contract_version: str = CONTRACT_VERSION
    cache_hit: bool = False

    def to_dict(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "status": self.status.value if isinstance(self.status, Status) else self.status,
            "answerable": bool(self.answerable),
            "error": self.error,
            "evidence": [item.to_dict() for item in self.evidence],
            "degraded_stages": list(self.degraded_stages),
            "latency_ms": dict(self.latency_ms),
            "index_version": self.index_version,
            "cache_hit": bool(self.cache_hit),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def error_result(
        cls,
        *,
        request_id: str,
        trace_id: str,
        index_version: str,
        code: ErrorCode,
        message: str,
        latency_ms: dict | None = None,
    ) -> "RetrievalResult":
        return cls(
            status=Status.ERROR,
            answerable=False,
            evidence=[],
            request_id=request_id,
            trace_id=trace_id,
            index_version=index_version,
            latency_ms=latency_ms or {},
            error={"code": code.value, "message": message},
        )


def validate_evidence_ids(answer: str, evidence_ids: list[str]) -> list[str]:
    """
    校验生成答案里的引用编号是否来自本次检索结果（v2 §3.1 输出引用校验）。

    支持两种写法：【参考资料N】与 [N]；返回“非法引用”列表，调用方负责记录告警。
    """
    import re

    allowed = set(evidence_ids)
    illegal: list[str] = []
    for match in re.finditer(r"【(ev_\d+)】|【参考资料(\d+)】|\[(\d+)\]", answer or ""):
        raw = match.group(1) or match.group(2) or match.group(3)
        if match.group(1):
            candidate = raw
        else:
            candidate = f"ev_{int(raw):03d}"
        if candidate not in allowed and raw not in allowed:
            illegal.append(match.group(0))
    return illegal
