"""
RAG 子 Agent 的检索平面：Dense + BM25 → RRF → Rerank → 父子扩展 → 证据组装。

调用契约见 `rag/contract.py`；本模块只负责“检索与证据”，不负责最终答案生成。
任何可降级的环节失败都不会拖垮整体：
- Dense 失败：只用 BM25；
- BM25 失败：只用 Dense；
- Rerank 失败/超时：退回 RRF 排序，并记录 `degraded_stages`；
- 两路都失败：返回 status=error, error.code=UPSTREAM_5XX。
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

import dashscope
from dashscope import TextReRank

from rag.answerability import AnswerabilityChecker
from rag.contract import ErrorCode, Evidence, RetrievalResult, Status
from rag.sparse_index import get_bm25_index
from rag.telemetry import TTLCache, Trace, make_cache_key
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_conf
from utils.logger_handler import logger

_RERANK_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-rerank")
_MAX_QUERY_CHARS = 2000


class RetrievalService:
    def __init__(self, vector_store: VectorStoreService | None = None, config: dict | None = None):
        self.vector_store = vector_store or VectorStoreService()
        full_conf = config or rag_conf
        self.cfg = full_conf["retrieval"]
        self.rerank_model = full_conf.get("rerank_model_name", "gte-rerank-v2")
        self.cache = TTLCache(
            max_size=int(self.cfg.get("cache_size", 128)),
            ttl_seconds=int(self.cfg.get("cache_ttl_seconds", 300)),
        )
        self.answerability = AnswerabilityChecker(
            timeout_ms=int(self.cfg.get("answerability_timeout_ms", 1500)),
            fallback_threshold=float(self.cfg.get("answerability_fallback_threshold", 0.15)),
            model_name=str(self.cfg.get("answerability_model_name", "qwen-flash")),
            high_threshold=float(self.cfg.get("answerability_high_threshold", 0.60)),
            low_threshold=float(self.cfg.get("answerability_low_threshold", 0.10)),
        )
        # 熔断状态：连续失败达到阈值后短时间跳过该阶段（v2 §2）
        self._rerank_failures = 0
        self._rerank_open_until = 0.0
        self._answerability_failures = 0
        self._answerability_open_until = 0.0

    # ------------------------------------------------------------------ public

    def retrieve(
        self,
        query: str,
        scope: dict | None = None,
        top_k: int | None = None,
        timeout_budget_ms: int | None = None,
    ) -> RetrievalResult:
        trace = Trace()
        top_k = int(top_k or self.cfg.get("final_top_n", 5))
        budget_ms = int(timeout_budget_ms or self.cfg.get("request_timeout_ms", 5000))

        invalid = self._validate_query(query)
        if invalid:
            result = RetrievalResult.error_result(
                request_id=trace.request_id,
                trace_id=trace.trace_id,
                index_version=self.vector_store.index_version,
                code=ErrorCode.INVALID_QUERY,
                message=invalid,
                latency_ms=trace.latency(),
            )
            trace.log(query=query or "", status=result.status.value, answerable=False, evidence=[])
            return result

        stale = self._index_stale_reason()
        if stale:
            result = RetrievalResult.error_result(
                request_id=trace.request_id,
                trace_id=trace.trace_id,
                index_version=self.vector_store.index_version,
                code=ErrorCode.INDEX_STALE,
                message=stale,
                latency_ms=trace.latency(),
            )
            trace.log(query=query, status=result.status.value, answerable=False, evidence=[])
            return result

        query = query.strip()
        cache_key = make_cache_key(
            {
                "query": query,
                "scope": scope or {},
                "top_k": top_k,
                "dense_k": self.cfg.get("dense_k", 15),
                "sparse_k": self.cfg.get("sparse_k", 15),
                "rrf_k": self.cfg.get("rrf_k", 60),
                "rerank_model": self.rerank_model,
                "index_version": self.vector_store.index_version,
            }
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._from_cache(cached, trace)

        dense_failed = False
        sparse_failed = False
        dense_candidates = []
        sparse_candidates = []

        with trace.stage("dense"):
            try:
                dense_candidates = self.vector_store.similarity_search_with_score(
                    query, k=int(self.cfg.get("dense_k", 15)), scope=scope
                )
            except Exception as exc:  # noqa: BLE001 - 降级路径必须可见
                dense_failed = True
                trace.mark_degraded("dense_error")
                logger.warning("[RAG] Dense 检索失败，降级到稀疏路：%s", exc)

        with trace.stage("sparse"):
            if self.cfg.get("enable_sparse", True):
                try:
                    documents = self.vector_store.get_all_documents()
                    bm25 = get_bm25_index(
                        f"{self.vector_store.index_version}:{len(documents)}", documents
                    )
                    allowed = self._allowed_ids(documents, scope) if scope else None
                    sparse_candidates = bm25.search(
                        query, k=int(self.cfg.get("sparse_k", 15)), allowed_ids=allowed
                    )
                except Exception as exc:  # noqa: BLE001
                    sparse_failed = True
                    trace.mark_degraded("sparse_error")
                    logger.warning("[RAG] BM25 检索失败，降级到向量路：%s", exc)

        if dense_failed and (sparse_failed or not self.cfg.get("enable_sparse", True)):
            result = RetrievalResult.error_result(
                request_id=trace.request_id,
                trace_id=trace.trace_id,
                index_version=self.vector_store.index_version,
                code=ErrorCode.UPSTREAM_5XX,
                message="dense 与 sparse 两路检索均失败",
                latency_ms=trace.latency(),
            )
            trace.log(query=query, status=result.status.value, answerable=False, evidence=[])
            return result

        with trace.stage("fuse"):
            fused = self._rrf_fuse(dense_candidates, sparse_candidates)

        ranked = fused
        rerank_used = False
        if ranked and self.cfg.get("enable_rerank", True):
            if time.monotonic() < self._rerank_open_until:
                trace.mark_degraded("rerank_circuit_open")
            else:
                with trace.stage("rerank"):
                    try:
                        ranked = self._rerank(query, ranked)
                        rerank_used = True
                        self._rerank_failures = 0
                    except FuturesTimeoutError:
                        self._rerank_failures += 1
                        trace.mark_degraded("rerank_timeout")
                        logger.warning("[RAG] 精排超时，退回 RRF 排序")
                    except Exception as exc:  # noqa: BLE001
                        self._rerank_failures += 1
                        trace.mark_degraded("rerank_error")
                        logger.warning("[RAG] 精排失败，退回 RRF 排序：%s", exc)
                    finally:
                        threshold = int(self.cfg.get("rerank_circuit_threshold", 3))
                        if self._rerank_failures >= threshold:
                            cooldown = float(self.cfg.get("rerank_cooldown_seconds", 120))
                            self._rerank_open_until = time.monotonic() + cooldown
                            logger.warning("[RAG] rerank 连续失败 %s 次，熔断 %.0fs", self._rerank_failures, cooldown)

        with trace.stage("expand"):
            evidence = self._assemble(ranked[:top_k], rerank_used)

        answerable = bool(evidence)
        if evidence:
            mode = str(self.cfg.get("answerability_mode", "llm")).lower()
            if mode == "llm":
                # 即使 rerank 降级，也仍然用 LLM 判断证据是否足以回答
                if time.monotonic() < self._answerability_open_until:
                    answerable = self._answerability_fallback(evidence)
                    trace.mark_degraded("answerability_circuit_open")
                else:
                    answerable, checker_degraded = self.answerability.check(query, evidence)
                    if checker_degraded:
                        self._answerability_failures += 1
                        trace.mark_degraded("answerability_fallback")
                        threshold = int(self.cfg.get("answerability_circuit_threshold", 3))
                        if self._answerability_failures >= threshold:
                            cooldown = float(self.cfg.get("answerability_cooldown_seconds", 120))
                            self._answerability_open_until = time.monotonic() + cooldown
                            logger.warning(
                                "[RAG] answerability 连续失败 %s 次，熔断 %.0fs",
                                self._answerability_failures,
                                cooldown,
                            )
                    else:
                        self._answerability_failures = 0
            elif rerank_used:
                threshold = float(self.cfg.get("answerability_threshold", 0.35))
                answerable = float(evidence[0].score.get("rerank") or 0.0) >= threshold

        if not evidence:
            status = Status.NO_EVIDENCE
        elif trace.degraded:
            status = Status.DEGRADED
        elif not answerable:
            status = Status.NO_EVIDENCE
        else:
            status = Status.OK

        latency = trace.latency()
        if latency.get("total", 0) > budget_ms:
            trace.mark_degraded("budget_exceeded")
            if status == Status.OK:
                status = Status.DEGRADED

        result = RetrievalResult(
            status=status,
            answerable=answerable,
            evidence=evidence,
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            index_version=self.vector_store.index_version,
            latency_ms=trace.latency(),
            degraded_stages=list(trace.degraded),
        )
        trace.log(query=query, status=status.value, answerable=answerable, evidence=evidence)
        self.cache.set(cache_key, result)
        return result

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _validate_query(query: str) -> str | None:
        if not isinstance(query, str) or not query.strip():
            return "query 不能为空"
        if len(query) > _MAX_QUERY_CHARS:
            return f"query 超过 {_MAX_QUERY_CHARS} 字符上限"
        return None

    def _index_stale_reason(self) -> str | None:
        manifest_version = self.vector_store.manifest.data.get("index_version")
        if manifest_version and manifest_version != self.vector_store.index_version:
            return (
                f"索引版本不一致：manifest={manifest_version}, "
                f"期望={self.vector_store.index_version}；请重建索引"
            )
        return None

    def _from_cache(self, cached: RetrievalResult, trace: Trace) -> RetrievalResult:
        result = RetrievalResult(
            status=cached.status,
            answerable=cached.answerable,
            evidence=cached.evidence,
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            index_version=cached.index_version,
            latency_ms=trace.latency(),
            degraded_stages=list(cached.degraded_stages),
            error=cached.error,
            cache_hit=True,
        )
        trace.log(
            query="(cache)", status=result.status.value, answerable=result.answerable, evidence=result.evidence
        )
        return result

    def _rrf_fuse(
        self,
        dense_candidates: list[tuple],
        sparse_candidates: list[tuple],
    ) -> list[dict]:
        """RRF：score = Σ 1/(k + rank)；不调权重，零配置稳健默认值。"""
        rrf_k = int(self.cfg.get("rrf_k", 60))
        merged: dict[str, dict] = {}

        def touch(document, chunk_id: str) -> dict:
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "doc": document,
                    "chunk_id": chunk_id,
                    "dense": 0.0,
                    "sparse": 0.0,
                    "rrf": 0.0,
                    "rerank": None,
                }
            return merged[chunk_id]

        for rank, (document, distance) in enumerate(dense_candidates, start=1):
            chunk_id = str(document.metadata.get("chunk_id") or "")
            if not chunk_id:
                continue
            item = touch(document, chunk_id)
            item["dense"] = round(1.0 / (1.0 + float(distance)), 6)
            item["rrf"] += 1.0 / (rrf_k + rank)

        for rank, (document, score) in enumerate(sparse_candidates, start=1):
            chunk_id = str(document.metadata.get("chunk_id") or "")
            if not chunk_id:
                continue
            item = touch(document, chunk_id)
            item["sparse"] = round(float(score), 6)
            item["rrf"] += 1.0 / (rrf_k + rank)

        return sorted(
            merged.values(),
            key=lambda item: (-item["rrf"], item["chunk_id"]),
        )

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        top_n = max(int(self.cfg.get("rerank_top_n", 5)), int(self.cfg.get("final_top_n", 5)))
        top_n = min(top_n, len(candidates))
        timeout_s = max(0.1, float(self.cfg.get("rerank_timeout_ms", 2000)) / 1000)

        future = _RERANK_POOL.submit(self._rerank_call, query, candidates, top_n)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            future.cancel()
            raise

    def _rerank_call(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        last_error: Exception | None = None
        response = None
        for attempt in range(2):
            try:
                response = TextReRank.call(
                    model=self.rerank_model,
                    query=query,
                    documents=[item["doc"].page_content for item in candidates],
                    top_n=top_n,
                    return_documents=False,
                )
                if getattr(response, "status_code", 0) == 200:
                    break
                last_error = RuntimeError(
                    f"rerank HTTP {getattr(response, 'status_code', '?')}: "
                    f"{getattr(response, 'message', '')}"
                )
                # 鉴权/额度类错误重试无意义，直接失败
                if getattr(response, "status_code", 0) in {401, 403}:
                    break
            except Exception as exc:  # noqa: BLE001 - 429/5xx 做一次短退避重试
                last_error = exc
            if attempt == 0:
                time.sleep(0.4)
        if response is None or getattr(response, "status_code", 0) != 200:
            raise last_error or RuntimeError("rerank failed")

        results = (response.output or {}).get("results") or []
        ranked: list[dict] = []
        for item in results:
            index = item.get("index")
            if not isinstance(index, int) or not (0 <= index < len(candidates)):
                continue
            candidate = dict(candidates[index])
            candidate["rerank"] = round(float(item.get("relevance_score", 0.0)), 6)
            ranked.append(candidate)

        ranked.sort(key=lambda item: (-float(item["rerank"] or 0.0), -item["rrf"]))
        return ranked

    def _assemble(self, candidates: list[dict], rerank_used: bool) -> list[Evidence]:
        max_chars = int(self.cfg.get("max_context_chars", 4000))
        parent_ids = [
            str(item["doc"].metadata.get("parent_id"))
            for item in candidates
            if item["doc"].metadata.get("parent_id")
        ]
        parents = self.vector_store.get_parents(parent_ids)

        evidence: list[Evidence] = []
        used_chars = 0
        for index, candidate in enumerate(candidates, start=1):
            document = candidate["doc"]
            metadata = document.metadata
            parent = parents.get(str(metadata.get("parent_id")))
            context = (parent or {}).get("text") or document.page_content
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            if len(context) > remaining:
                context = context[:remaining]
            used_chars += len(context)

            title_path = metadata.get("title_path")
            if isinstance(title_path, str):
                try:
                    title_path = json.loads(title_path)
                except json.JSONDecodeError:
                    title_path = [title_path]
            scope_tags = metadata.get("scope_tags")
            if isinstance(scope_tags, str):
                try:
                    scope_tags = json.loads(scope_tags)
                except json.JSONDecodeError:
                    scope_tags = [scope_tags]

            evidence.append(
                Evidence(
                    evidence_id=f"ev_{index:03d}",
                    doc_id=str(metadata.get("doc_id", "")),
                    version=str(metadata.get("version", "")),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    parent_id=str(metadata.get("parent_id", "")),
                    title_path=title_path or [],
                    page=int(metadata.get("page", 0) or 0),
                    char_start=int(metadata.get("char_start", 0) or 0),
                    char_end=int(metadata.get("char_end", 0) or 0),
                    text=document.page_content,
                    context=context,
                    source=str(metadata.get("source", "")),
                    scope_tags=scope_tags or [],
                    score={
                        "dense": candidate.get("dense", 0.0),
                        "sparse": candidate.get("sparse", 0.0),
                        "rrf": round(float(candidate.get("rrf", 0.0)), 6),
                        "rerank": candidate.get("rerank") if rerank_used else None,
                    },
                )
            )
        return evidence

    @staticmethod
    def _answerability_fallback(evidence: list) -> bool:
        """LLM 熔断/失败时的保守兜底：没有 rerank 分数或分数不足都按不可回答处理。"""
        if not evidence:
            return False
        top = evidence[0].score.get("rerank")
        if top is None:
            return False
        return float(top) >= 0.35

    @staticmethod
    def _allowed_ids(documents: list, scope: dict) -> set[str]:
        """BM25 的召回前过滤：只允许 scope 命中的块进入打分。"""
        allowed: set[str] = set()
        for document in documents:
            metadata = document.metadata or {}
            matched = True
            for key, expected in scope.items():
                value = metadata.get(key)
                if isinstance(expected, (list, tuple)):
                    expected = json.dumps(list(expected), ensure_ascii=False)
                if str(value) != str(expected):
                    matched = False
                    break
            if matched:
                chunk_id = metadata.get("chunk_id")
                if chunk_id:
                    allowed.add(str(chunk_id))
        return allowed
