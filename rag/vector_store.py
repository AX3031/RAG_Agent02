"""
向量库服务：结构化入库 + 索引清单 + 父子块存储。

与旧实现的差异（对应 v2 §4.1 / §4.2 / §4.4）：
1. 每个块写入确定性 chunk_id 与完整元数据，而不是只依赖 Chroma 自动 UUID。
2. 文件内容变化时：先写新版本，再按清单删除旧版本块（版本传播），不再只增不删。
3. 文件从磁盘消失时可选择 prune，把它的块与父块一起清理。
4. 父块单独持久化，检索命中子块后可按 parent_id 扩展上下文。
5. `md5.text` 不再作为权威去重依据；清单 `storage/index_manifest.json` 取代它。
"""

from __future__ import annotations

import argparse
import os
import threading

from langchain_chroma import Chroma
from langchain_core.documents import Document

from model.factory import embed_model
from rag.chunking import compute_index_version, split_documents
from rag.index_store import IndexManifest, ParentStore
from utils.config_handler import chroma_conf, rag_conf
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path, get_project_root


class VectorStoreService:
    def __init__(self):
        self.index_version = compute_index_version(
            embedding_model=rag_conf["embedding_model_name"],
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            parent_size=chroma_conf["parent_chunk_size"],
        )
        self.collection_name = chroma_conf["collection_name"]
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )
        self.manifest = IndexManifest(
            get_abs_path(chroma_conf["index_manifest"]), self.index_version
        )
        self.parent_store = ParentStore(
            get_abs_path(chroma_conf["parent_store"]), self.index_version
        )
        self._lock = threading.RLock()

    # ---------- 基础查询 ----------

    def get_retriever(self, k: int | None = None):
        """保留旧接口：纯向量检索器（新链路请用 similarity_search_with_score）。"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": k or chroma_conf["k"]}
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        scope: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """返回 (Document, distance)。Chroma 默认 L2 距离，越小越相似。"""
        where = self._build_where(scope)
        kwargs = {"k": k}
        if where:
            kwargs["filter"] = where
        return self.vector_store.similarity_search_with_score(query, **kwargs)

    def get_all_documents(self) -> list[Document]:
        """读取集合内全部块，供 BM25 索引构建。"""
        result = self.vector_store.get(include=["documents", "metadatas"])
        documents: list[Document] = []
        for text, metadata in zip(result.get("documents") or [], result.get("metadatas") or []):
            if text is None:
                continue
            documents.append(Document(page_content=text, metadata=dict(metadata or {})))
        return documents

    def get_parents(self, parent_ids: list[str]) -> dict[str, dict]:
        return self.parent_store.get_many(parent_ids)

    def stats(self) -> dict:
        try:
            count = self.vector_store._collection.count()  # noqa: SLF001 - langchain-chroma 无公开计数 API
        except Exception:
            count = -1
        return {
            "collection": self.collection_name,
            "index_version": self.index_version,
            "chunks": count,
            "files": len(self.manifest.all_files()),
            "parents": len(self.parent_store.all_parents()),
        }

    # ---------- 入库 / 更新 / 删除 ----------

    def load_document(self, prune: bool = True) -> dict:
        """
        增量入库：
        - 新文件：切分、写父块、写子块、登记清单；
        - 已存在且 MD5 相同：跳过；
        - 已存在但 MD5 变化：先写新版本，再删除旧版本块；
        - prune=True 时，清理磁盘上已不存在的文件。
        """
        report = {"indexed": [], "skipped": [], "failed": [], "pruned": []}
        data_root = get_abs_path(chroma_conf["data_path"])
        allowed = tuple(chroma_conf["allow_knowledge_file_type"])
        paths = listdir_with_allowed_type(data_root, allowed)

        with self._lock:
            for abs_path in paths:
                rel_path = os.path.relpath(abs_path, get_project_root()).replace("\\", "/")
                md5_hex = get_file_md5_hex(abs_path)
                if not md5_hex:
                    report["failed"].append({"file": rel_path, "error": "md5 计算失败"})
                    continue

                existing = self.manifest.get_file(rel_path)
                if existing and existing.get("version") == md5_hex:
                    report["skipped"].append(rel_path)
                    continue

                try:
                    documents = self._load_file(abs_path)
                    if not documents:
                        logger.warning(f"【加载知识库】{rel_path} 没有有效文本内容，跳过")
                        report["failed"].append({"file": rel_path, "error": "无有效文本"})
                        continue

                    bundle = split_documents(
                        source_path=abs_path,
                        rel_path=rel_path,
                        documents=documents,
                        index_version=self.index_version,
                        embedding_model=rag_conf["embedding_model_name"],
                        chunk_size=chroma_conf["chunk_size"],
                        chunk_overlap=chroma_conf["chunk_overlap"],
                        parent_size=chroma_conf["parent_chunk_size"],
                        separators=chroma_conf["separators"],
                        scope_tags=chroma_conf.get("scope_tags"),
                    )
                    if not bundle.child_docs:
                        logger.warning(f"【加载知识库】{rel_path} 分片后没有有效内容，跳过")
                        report["failed"].append({"file": rel_path, "error": "分片为空"})
                        continue

                    new_ids = [doc.metadata["chunk_id"] for doc in bundle.child_docs]
                    self.vector_store.add_documents(bundle.child_docs, ids=new_ids)
                    self.parent_store.upsert_document(bundle.doc_id, bundle.parents)

                    # 版本传播：新版本落库成功后再删除旧版本，避免中途失败丢数据
                    if existing:
                        stale_ids = [cid for cid in existing.get("chunk_ids", []) if cid not in set(new_ids)]
                        self._delete_ids(stale_ids)

                    self.manifest.upsert_file(
                        rel_path,
                        {
                            "doc_id": bundle.doc_id,
                            "version": bundle.version,
                            "chunk_ids": new_ids,
                            "parent_ids": sorted(bundle.parents.keys()),
                            "chunk_count": len(new_ids),
                            "parent_count": len(bundle.parents),
                            "index_version": self.index_version,
                        },
                    )
                    report["indexed"].append(
                        {
                            "file": rel_path,
                            "doc_id": bundle.doc_id,
                            "version": bundle.version,
                            **bundle.stats,
                        }
                    )
                    logger.info(
                        "【加载知识库】%s 完成：%s 个父块 / %s 个子块（index_version=%s）",
                        rel_path,
                        bundle.stats["parents"],
                        bundle.stats["children"],
                        self.index_version,
                    )
                except Exception as exc:  # noqa: BLE001 - 单文件失败不影响其他文件
                    logger.error(f"【加载知识库】{rel_path} 加载失败：{exc}", exc_info=True)
                    report["failed"].append({"file": rel_path, "error": str(exc)})

            if prune:
                existing_files = {
                    os.path.relpath(p, get_project_root()).replace("\\", "/") for p in paths
                }
                for rel_path, record in self.manifest.all_files().items():
                    if rel_path in existing_files:
                        continue
                    self._delete_ids(record.get("chunk_ids", []))
                    self.parent_store.remove_document(record.get("doc_id", ""))
                    self.manifest.remove_file(rel_path)
                    report["pruned"].append(rel_path)
                    logger.info("【加载知识库】%s 已从磁盘移除，清理其索引", rel_path)

        return report

    def delete_document(self, doc_id: str) -> bool:
        """按 doc_id 级联删除子块、父块与清单记录（v2 §3.4 删除权的最小实现）。"""
        with self._lock:
            for rel_path, record in self.manifest.all_files().items():
                if record.get("doc_id") != doc_id:
                    continue
                self._delete_ids(record.get("chunk_ids", []))
                self.parent_store.remove_document(doc_id)
                self.manifest.remove_file(rel_path)
                logger.info("【删除文档】doc_id=%s（%s）已删除", doc_id, rel_path)
                return True
        return False

    def rebuild(self) -> dict:
        """全量重建：清空当前集合与父块，再从磁盘重新入库。"""
        with self._lock:
            ids: list[str] = []
            for record in self.manifest.all_files().values():
                ids.extend(record.get("chunk_ids", []))
            self._delete_ids(ids)
            # 兜底：清理集合中不在清单里的历史块
            try:
                remaining = self.get_all_documents()
                extra_ids = [
                    doc.metadata.get("chunk_id")
                    for doc in remaining
                    if doc.metadata.get("chunk_id") and doc.metadata["chunk_id"] not in set(ids)
                ]
                self._delete_ids([i for i in extra_ids if i])
            except Exception as exc:  # noqa: BLE001
                logger.warning("【重建索引】清理历史块失败：%s", exc)
            self.manifest.reset()
            self.parent_store.reset()
        return self.load_document(prune=True)

    # ---------- 内部工具 ----------

    def _load_file(self, abs_path: str) -> list[Document]:
        suffix = abs_path.lower().rsplit(".", 1)[-1]
        if suffix == "txt":
            return txt_loader(abs_path)
        if suffix == "pdf":
            return pdf_loader(abs_path)
        return []

    def _delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self.vector_store.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001 - 删除失败要可见但不阻断整体流程
            logger.warning("【删除向量】%s 个块删除失败：%s", len(ids), exc)

    @staticmethod
    def _build_where(scope: dict | None) -> dict | None:
        """把 {metadata_key: value} 转成 Chroma 的 where 条件。

        约定：list/tuple 值按 JSON 字符串精确匹配（scope_tags 在入库时也是 JSON 字符串）。
        """
        if not scope:
            return None
        import json

        clauses = []
        for key, value in scope.items():
            if isinstance(value, (list, tuple)):
                value = json.dumps(list(value), ensure_ascii=False)
            clauses.append({key: value})
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库入库 / 重建 / 删除")
    parser.add_argument("--rebuild", action="store_true", help="清空当前集合并全量重建")
    parser.add_argument("--no-prune", action="store_true", help="不清理磁盘上已删除文件对应的索引")
    parser.add_argument("--delete-doc", default=None, help="按 doc_id 删除一篇文档")
    args = parser.parse_args()

    service = VectorStoreService()
    if args.delete_doc:
        ok = service.delete_document(args.delete_doc)
        print("deleted" if ok else "not found")
        return

    report = service.rebuild() if args.rebuild else service.load_document(prune=not args.no_prune)
    print(report)
    print(service.stats())


if __name__ == "__main__":
    main()
