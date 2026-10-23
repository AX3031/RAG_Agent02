"""
可回答性（Answerability）判定。

为什么不能只靠 rerank 阈值：
- 实测发现「主题相关但事实缺失」的负样本（例如问《论语》里有没有股票投资）
  与真正可回答的样本，top rerank 分数区间高度重叠；
- 任何单一阈值都会在“拒答率”和“可回答率”之间二选一。

因此实现两层：
1. 轻量 LLM 判定：只看证据能否支撑答案，要求只输出 YES/NO；
2. 阈值兜底：LLM 超时/失败时退回 rerank 分数阈值，并标记 degraded。
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from model.factory import chat_model
from rag.telemetry import TTLCache, make_cache_key, hash_text
from utils.logger_handler import logger

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-answerability")

_SYSTEM_PROMPT = """你是企业 RAG 子 Agent 的可回答性检查器。你的唯一任务是判断：仅凭给定证据，能否回答用户问题。

判定 YES 的条件：证据中明确包含回答问题所需的关键事实（人物、事件、数字、定义、步骤等），不需要任何外部知识补充。
判定 NO 的情况：
- 证据只与问题主题相关，但没有给出问题所问的具体答案；
- 证据提到的对象/属性与问题所问不匹配；
- 证据不足以支撑一个确定答案。

不要因为证据里出现了问题中的关键词就判 YES。
只输出一个词：YES 或 NO。"""


class AnswerabilityChecker:
    def __init__(
        self,
        timeout_ms: int = 2000,
        fallback_threshold: float = 0.15,
        model_name: str = "qwen-flash",
        high_threshold: float = 0.60,
        low_threshold: float = 0.10,
    ):
        self.timeout_ms = max(100, int(timeout_ms))
        self.fallback_threshold = float(fallback_threshold)
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.cache = TTLCache(max_size=512, ttl_seconds=600)
        self.model = self._build_model(model_name)

    @staticmethod
    def _build_model(model_name: str):
        """可回答性判定用更快的模型（qwen-flash），避免拖慢检索主链路。"""
        try:
            return init_chat_model(
                model=model_name,
                model_provider="openai",
                api_key=os.getenv("OPEN_API_KEY"),
                base_url=os.getenv("BASE_URL_DASHSCOPE"),
            )
        except Exception as exc:  # noqa: BLE001 - 模型构造失败时退回主模型
            logger.warning("[RAG] Answerability 模型初始化失败，改用主模型：%s", exc)
            return chat_model

    def check(self, query: str, evidence: list) -> tuple[bool, bool]:
        """
        返回 (answerable, degraded)。
        degraded=True 表示 LLM 判定失败、使用了阈值兜底。
        """
        if not evidence:
            return False, False

        # 快速路径：分数极端时不必调用 LLM，降低延迟与成本
        top = evidence[0].score.get("rerank")
        if top is not None:
            if float(top) >= self.high_threshold:
                return True, False
            if float(top) < self.low_threshold:
                return False, False

        cache_key = make_cache_key(
            {"q": hash_text(query), "ev": [item.chunk_id for item in evidence]}
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return bool(cached), False

        try:
            future = _POOL.submit(self._ask_llm, query, evidence)
            answerable = future.result(timeout=self.timeout_ms / 1000)
            self.cache.set(cache_key, answerable)
            return answerable, False
        except FuturesTimeoutError:
            future.cancel()
            logger.warning("[RAG] Answerability 判定超时，使用阈值兜底")
        except Exception as exc:  # noqa: BLE001 - 兜底路径必须可见
            logger.warning("[RAG] Answerability 判定失败，使用阈值兜底：%s", exc)

        return self._fallback(evidence), True

    def _ask_llm(self, query: str, evidence: list) -> bool:
        blocks = []
        used = 0
        for item in evidence:
            text = item.context or item.text
            if used + len(text) > 3000:
                text = text[: max(0, 3000 - used)]
            if not text:
                break
            used += len(text)
            blocks.append(f"[{item.evidence_id}]\n{text}")
        prompt = "用户问题：\n" + query + "\n\n证据：\n" + "\n\n".join(blocks) + "\n\n只输出 YES 或 NO。"
        message = self.model.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        content = message.content if isinstance(message.content, str) else str(message.content)
        return self._parse(content)

    @staticmethod
    def _parse(text: str) -> bool:
        upper = (text or "").strip().upper()
        match = re.search(r"\b(YES|NO)\b", upper)
        if match:
            return match.group(1) == "YES"
        logger.warning("[RAG] Answerability 输出无法解析，按 NO 处理")
        return False

    def _fallback(self, evidence: list) -> bool:
        top = evidence[0].score.get("rerank") if evidence else None
        if top is None:
            # 没有精排分数时无法确认相关性，保守判为不可回答
            return False
        return float(top) >= self.fallback_threshold
