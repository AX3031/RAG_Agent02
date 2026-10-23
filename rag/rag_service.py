"""
RAG 受控生成服务。

职责划分（v2 §1 定位）：
- `retrieve()` 返回结构化证据（由 `RetrievalService` 完成）；
- `rag_summarize()` 只负责“基于证据的受控生成”，并把证据作为不可信数据包裹；
- 最终答案的引用必须来自本次 `evidence[]`，非法引用会记录告警。
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from rag.contract import RetrievalResult, Status, validate_evidence_ids
from rag.retrieval_service import RetrievalService
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


class RagSummarizeService:
    def __init__(self, retrieval: RetrievalService | None = None):
        self.retrieval = retrieval or RetrievalService()
        self.prompt_template = PromptTemplate.from_template(load_rag_prompts())
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | self.model | StrOutputParser()

    # ---------- 子 Agent 契约 ----------

    def retrieve(self, query: str, scope: dict | None = None, top_k: int | None = None) -> RetrievalResult:
        return self.retrieval.retrieve(query, scope=scope, top_k=top_k)

    # ---------- 兼容旧工具：受控生成 ----------

    def retriever_docs(self, query: str):
        """兼容旧接口：返回证据对应的 Document 列表（新代码请直接使用 retrieve）。"""
        result = self.retrieve(query)
        return [self._evidence_to_document(item) for item in result.evidence]

    def rag_summarize(self, query: str) -> str:
        result = self.retrieve(query)

        if result.status == Status.ERROR:
            code = (result.error or {}).get("code", "UPSTREAM_5XX")
            logger.warning("[RAG] 检索失败 request_id=%s code=%s", result.request_id, code)
            return f"检索失败（{code}），请稍后重试。"

        if not result.answerable or not result.evidence:
            return "根据现有资料无法回答该问题。"

        context = self._format_context(result)
        answer = self.chain.invoke({"input": query, "context": context})

        illegal = validate_evidence_ids(answer, [item.evidence_id for item in result.evidence])
        if illegal:
            logger.warning(
                "[RAG] 生成答案包含非法引用 %s（request_id=%s，允许的 evidence_id=%s）",
                illegal,
                result.request_id,
                [item.evidence_id for item in result.evidence],
            )
        return answer

    # ---------- 内部 ----------

    @staticmethod
    def _format_context(result: RetrievalResult) -> str:
        blocks = []
        for item in result.evidence:
            title = " / ".join(item.title_path) if item.title_path else item.doc_id
            blocks.append(
                "\n".join(
                    [
                        f"【参考资料{item.evidence_id}｜来源={item.source}｜"
                        f"标题={title}｜页码={item.page + 1}】",
                        "<<<EVIDENCE",
                        item.context,
                        "EVIDENCE>>>",
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _evidence_to_document(item):
        from langchain_core.documents import Document

        return Document(page_content=item.text, metadata=item.to_dict())


if __name__ == "__main__":
    service = RagSummarizeService()
    print(service.rag_summarize("学而时习之是什么意思"))
