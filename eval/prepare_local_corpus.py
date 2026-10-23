"""
把本地《对RAG的初步要求.docx》转换成可入库的纯文本语料。

为什么单独写脚本：
- 该文档是内部需求文档，`.docx` 与转换后的 `.txt` 都不进入版本管理；
- 转换过程不依赖 python-docx（当前环境未安装），直接解包 docx 读取 `word/document.xml`；
- 保留原有的 `##` / `###` 标题层级，结构化分块才能识别出 title_path。

用法：
    uv run -m eval.prepare_local_corpus
    # 或
    .venv\\Scripts\\python.exe -m eval.prepare_local_corpus
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCX_PATH = PROJECT_ROOT / "对RAG的初步要求.docx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "rag_requirements.txt"

# docx 表格里的分隔行、纯页码行没有检索价值，转换时丢弃
_TABLE_SEP = re.compile(r"^\|?[\s:|-]+\|?$")
_PAGE_NO = re.compile(r"^\s*\d+\s*$")


def extract_paragraphs(docx_path: Path) -> list[str]:
    with zipfile.ZipFile(docx_path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
    texts = []
    for paragraph in paragraphs:
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", paragraph, re.S))
        text = (
            text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
        ).strip()
        if not text or _TABLE_SEP.match(text) or _PAGE_NO.match(text):
            continue
        texts.append(text)
    return texts


def main() -> None:
    if not DOCX_PATH.exists():
        raise SystemExit(f"找不到需求文档：{DOCX_PATH}")
    paragraphs = extract_paragraphs(DOCX_PATH)
    OUTPUT_PATH.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    print(f"已生成 {OUTPUT_PATH}：{len(paragraphs)} 段，{OUTPUT_PATH.stat().st_size} 字节")


if __name__ == "__main__":
    main()
