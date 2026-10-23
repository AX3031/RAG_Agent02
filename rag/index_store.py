"""
索引清单（manifest）与父块存储。

为什么需要它（v2 §4.2 / §4.4）：
- 旧的 `md5.text` 只能回答“这个 MD5 处理过没有”，无法回答“它对应哪些块”，
  因此文档更新后旧块无法删除，知识库只会越滚越大（TD-07）。
- 这里把「文件 -> 版本 -> chunk_id 列表 / parent_id 列表」持久化，
  删除与更新都能按 ID 精确传播；写入使用临时文件 + os.replace 保证原子性。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: str, data: dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path_obj)


class IndexManifest:
    """文件级索引清单：记录每个已入库文件的版本与块 ID，支持增量更新与删除传播。"""

    def __init__(self, path: str, index_version: str):
        self.path = path
        self.index_version = index_version
        self._lock = threading.RLock()
        self.data = {"index_version": index_version, "updated_at": None, "files": {}}
        self.load()

    def load(self) -> None:
        with self._lock:
            if not os.path.exists(self.path):
                return
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data = loaded
                    self.data.setdefault("files", {})
            except (json.JSONDecodeError, OSError):
                # 清单损坏时不要静默丢弃：保留备份，后续由 rebuild 重建
                backup = self.path + ".corrupt"
                try:
                    os.replace(self.path, backup)
                except OSError:
                    pass
                self.data = {"index_version": self.index_version, "updated_at": None, "files": {}}

    def get_file(self, rel_path: str) -> dict | None:
        return self.data.get("files", {}).get(rel_path.replace("\\", "/"))

    def upsert_file(self, rel_path: str, record: dict) -> None:
        with self._lock:
            key = rel_path.replace("\\", "/")
            record = dict(record)
            record.setdefault("indexed_at", _now_iso())
            self.data.setdefault("files", {})[key] = record
            self.data["index_version"] = self.index_version
            self.data["updated_at"] = _now_iso()
            _atomic_write_json(self.path, self.data)

    def remove_file(self, rel_path: str) -> dict | None:
        with self._lock:
            key = rel_path.replace("\\", "/")
            record = self.data.get("files", {}).pop(key, None)
            if record is not None:
                self.data["updated_at"] = _now_iso()
                _atomic_write_json(self.path, self.data)
            return record

    def all_files(self) -> dict:
        return dict(self.data.get("files", {}))

    def reset(self) -> None:
        with self._lock:
            self.data = {"index_version": self.index_version, "updated_at": _now_iso(), "files": {}}
            _atomic_write_json(self.path, self.data)

    def save(self) -> None:
        with self._lock:
            self.data["index_version"] = self.index_version
            self.data["updated_at"] = _now_iso()
            _atomic_write_json(self.path, self.data)


class ParentStore:
    """父块存储：按 parent_id 取回完整上下文，供精排后的父子扩展使用。"""

    def __init__(self, path: str, index_version: str):
        self.path = path
        self.index_version = index_version
        self._lock = threading.RLock()
        self.data = {"index_version": index_version, "parents": {}}
        self.load()

    def load(self) -> None:
        with self._lock:
            if not os.path.exists(self.path):
                return
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data = loaded
                    self.data.setdefault("parents", {})
            except (json.JSONDecodeError, OSError):
                self.data = {"index_version": self.index_version, "parents": {}}

    def get(self, parent_id: str) -> dict | None:
        return self.data.get("parents", {}).get(parent_id)

    def get_many(self, parent_ids: list[str]) -> dict[str, dict]:
        parents = self.data.get("parents", {})
        return {pid: parents[pid] for pid in parent_ids if pid in parents}

    def upsert_document(self, doc_id: str, parents: dict[str, dict]) -> None:
        """整篇文档覆盖写入：先移除该文档旧父块，避免版本切换后父子悬空。"""
        with self._lock:
            store = self.data.setdefault("parents", {})
            for pid in [k for k, v in store.items() if v.get("doc_id") == doc_id]:
                store.pop(pid, None)
            store.update(parents)
            self.data["index_version"] = self.index_version
            _atomic_write_json(self.path, self.data)

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            store = self.data.setdefault("parents", {})
            for pid in [k for k, v in store.items() if v.get("doc_id") == doc_id]:
                store.pop(pid, None)
            _atomic_write_json(self.path, self.data)

    def all_parents(self) -> dict:
        return dict(self.data.get("parents", {}))

    def reset(self) -> None:
        with self._lock:
            self.data = {"index_version": self.index_version, "parents": {}}
            _atomic_write_json(self.path, self.data)
