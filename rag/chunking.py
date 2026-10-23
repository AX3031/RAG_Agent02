"""
结构化分块：把原始文档切成「父块 + 子块」，并给每个块写入稳定、可追溯的元数据。

设计目标（对应《对RAG的初步要求》v2 §4.1 / §4.5）：
1. 子块用于精确召回，父块用于上下文扩展（父子分块）。
2. 块 ID 是确定性的：同一文件 + 同一套分块参数 -> 同一批 chunk_id，评测标注可复用。
3. 每个子块携带 doc_id / version / chunk_id / parent_id / page / title_path 等字段。
4. 分块过程不依赖网络，可离线重复执行。

这里刻意不引入 jieba/tiktoken 等新依赖：
- 中文按「结构（标题/对话行/空行）」切分，再按字符长度控制块大小；
- token 统计由调用方按需换算（v2 §4.5 的 tokenizer 要求留给后续在配置中校准）。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKER_VERSION = "structure-v1"

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_JUAN_HEADING = re.compile(r"^卷[一二三四五六七八九十百]+[\s\u3000]+[\u4e00-\u9fff]{1,12}$")
_CHAPTER_HEADING = re.compile(r"^第[一二三四五六七八九十百0-9]+[章节篇部]")
# 《论语》中的对话行，每一段「子曰/有子曰/某某问」都从新块开始
_DIALOGUE = re.compile(r"^[\u4e00-\u9fff]{1,6}(?:曰|问|对曰)[：:]")


def stable_doc_id(rel_path: str) -> str:
    """由相对路径生成稳定、可读的 doc_id；同名文件追加路径哈希避免冲突。"""
    rel = rel_path.replace("\\", "/")
    stem = Path(rel).stem.strip() or "document"
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:6]
    return f"{stem}-{digest}"


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def compute_index_version(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    parent_size: int,
) -> str:
    """索引版本 = 分块算法 + 分块参数 + Embedding 模型；任一项变化都必须重建索引。"""
    raw = f"{CHUNKER_VERSION}|{embedding_model}|{chunk_size}|{chunk_overlap}|{parent_size}"
    return "idx-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


@dataclass
class Block:
    text: str
    page: int
    start: int
    end: int
    heading: tuple[int, str] | None = None


@dataclass
class ParentRecord:
    parent_id: str
    doc_id: str
    version: str
    source: str
    page: int
    char_start: int
    char_end: int
    title_path: list[str]
    text: str
    content_hash: str


@dataclass
class ChunkBundle:
    doc_id: str
    version: str
    rel_path: str
    child_docs: list[Document] = field(default_factory=list)
    parents: dict[str, dict] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def detect_heading(line: str) -> tuple[int, str] | None:
    """识别 Markdown 标题与《论语》卷名；返回 (层级, 标题)。"""
    text = line.strip()
    md = _MD_HEADING.match(text)
    if md:
        return len(md.group(1)), md.group(2).strip()
    if _JUAN_HEADING.match(text):
        return 1, text
    if _CHAPTER_HEADING.match(text) and len(text) <= 24:
        return 2, text
    return None


def _page_blocks(page_text: str, page_no: int) -> list[Block]:
    """把一页文本切成块：标题、对话行、空行都是边界。"""
    blocks: list[Block] = []
    buf: list[str] = []
    buf_start: int | None = None

    def flush(end_offset: int) -> None:
        nonlocal buf, buf_start
        if buf:
            text = "\n".join(buf).strip()
            if text and buf_start is not None:
                blocks.append(Block(text, page_no, buf_start, end_offset))
        buf = []
        buf_start = None

    offset = 0
    for raw_line in page_text.splitlines(keepends=True):
        line_start = offset
        offset += len(raw_line)
        stripped = raw_line.strip()

        if not stripped:
            flush(line_start)
            continue

        heading = detect_heading(stripped)
        if heading is not None:
            flush(line_start)
            blocks.append(
                Block(stripped, page_no, line_start, line_start + len(stripped), heading=heading)
            )
            continue

        if _DIALOGUE.match(stripped) and buf:
            flush(line_start)

        if buf_start is None:
            buf_start = line_start
        buf.append(stripped)

        # 防御性上限：单行/粘连文本过长时尽快落块
        if sum(len(part) for part in buf) > 1200:
            flush(offset)

    flush(offset)
    return blocks


def _build_parents(
    blocks: list[Block],
    title_state: list[str],
    parent_size: int,
) -> list[dict]:
    """把块按标题层级与长度聚合成父块。"""
    parents: list[dict] = []
    current: list[str] = []
    current_len = 0
    current_page = 0
    current_start = 0
    current_end = 0
    current_title: list[str] = []
    current_has_body = False

    def flush() -> None:
        nonlocal current, current_len, current_page, current_start, current_end, current_title, current_has_body
        if current:
            text = "\n".join(current).strip()
            if text:
                parents.append(
                    {
                        "text": text,
                        "page": current_page,
                        "char_start": current_start,
                        "char_end": current_end,
                        "title_path": list(current_title) or ["(root)"],
                    }
                )
        current = []
        current_len = 0
        current_title = []
        current_has_body = False

    for block in blocks:
        if block.heading is not None:
            # 连续标题合并到同一个父块，避免产生只有标题的碎片块
            if current and current_has_body:
                flush()
            level, title = block.heading
            while len(title_state) >= level:
                title_state.pop()
            title_state.append(title)
            if not current:
                current_title = list(title_state)
                current_page = block.page
                current_start = block.start
            current.append(block.text)
            current_len += len(block.text) + 1
            current_end = block.end
            continue

        if current and current_len + len(block.text) + 1 > parent_size:
            flush()

        if not current:
            current_page = block.page
            current_start = block.start
            current_title = list(title_state) or ["(root)"]

        current.append(block.text)
        current_len += len(block.text) + 1
        current_end = block.end
        current_has_body = True

    flush()
    return parents


def split_documents(
    source_path: str,
    rel_path: str,
    documents: list[Document],
    *,
    index_version: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    parent_size: int,
    separators: list[str],
    scope_tags: list[str] | None = None,
) -> ChunkBundle:
    """
    把 Loader 产出的页面级 Document 切成父子块。

    documents 的 metadata 约定：`page` 为 0-based 页码（PyPDFLoader 默认如此），
    `source` 为原始文件路径（会统一改写为 rel_path）。
    """
    rel = rel_path.replace("\\", "/")
    doc_id = stable_doc_id(rel)
    version = hashlib.md5(Path(source_path).read_bytes()).hexdigest()

    scope = json.dumps(scope_tags or ["public"], ensure_ascii=False)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        keep_separator=True,
    )

    bundle = ChunkBundle(doc_id=doc_id, version=version, rel_path=rel)
    title_state: list[str] = []
    child_seq = 0
    parent_seq = 0
    total_chars = 0

    for document in documents:
        page_no = int(document.metadata.get("page", 0) or 0)
        page_text = document.page_content or ""
        total_chars += len(page_text)

        blocks = _page_blocks(page_text, page_no)
        for parent in _build_parents(blocks, title_state, parent_size):
            parent_id = f"{doc_id}#{version[:8]}#p{parent_seq:04d}"
            parent_seq += 1

            parent_text = parent["text"]
            record = ParentRecord(
                parent_id=parent_id,
                doc_id=doc_id,
                version=version,
                source=rel,
                page=parent["page"],
                char_start=parent["char_start"],
                char_end=parent["char_end"],
                title_path=parent["title_path"],
                text=parent_text,
                content_hash=content_hash(parent_text),
            )
            bundle.parents[parent_id] = {
                "parent_id": record.parent_id,
                "doc_id": record.doc_id,
                "version": record.version,
                "source": record.source,
                "page": record.page,
                "char_start": record.char_start,
                "char_end": record.char_end,
                "title_path": record.title_path,
                "text": record.text,
                "content_hash": record.content_hash,
            }

            cursor = 0
            for piece in splitter.split_text(parent_text):
                text = piece.strip()
                if not text:
                    continue
                offset = parent_text.find(text, cursor)
                if offset < 0:
                    offset = cursor
                cursor = offset + len(text)

                chunk_id = f"{doc_id}#{version[:8]}#c{child_seq:04d}"
                child_seq += 1
                metadata = {
                    "doc_id": doc_id,
                    "version": version,
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "seq": child_seq - 1,
                    "source": rel,
                    "page": int(parent["page"]),
                    "char_start": int(parent["char_start"] + offset),
                    "char_end": int(parent["char_start"] + offset + len(text)),
                    "title_path": json.dumps(parent["title_path"], ensure_ascii=False),
                    "content_hash": content_hash(text),
                    "scope_tags": scope,
                    "embedding_model": embedding_model,
                    "index_version": index_version,
                    "is_active": True,
                }
                bundle.child_docs.append(Document(page_content=text, metadata=metadata))

    bundle.stats = {
        "doc_id": doc_id,
        "version": version,
        "pages": len(documents),
        "chars": total_chars,
        "parents": len(bundle.parents),
        "children": len(bundle.child_docs),
    }
    return bundle
