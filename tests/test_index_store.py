import json
import os
import tempfile
import unittest

from rag.index_store import IndexManifest, ParentStore


class IndexStoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        self.parent_path = os.path.join(self.dir, "parents.json")

    def test_manifest_upsert_and_remove(self):
        manifest = IndexManifest(self.manifest_path, "idx-1")
        manifest.upsert_file("data/a.txt", {"doc_id": "a", "version": "v1", "chunk_ids": ["c1"]})
        manifest.upsert_file("data/b.txt", {"doc_id": "b", "version": "v1", "chunk_ids": ["c2"]})
        self.assertEqual(len(manifest.all_files()), 2)

        record = manifest.remove_file("data/a.txt")
        self.assertEqual(record["doc_id"], "a")
        self.assertEqual(len(manifest.all_files()), 1)

        # 重新加载后仍然持久化
        reloaded = IndexManifest(self.manifest_path, "idx-1")
        self.assertIsNone(reloaded.get_file("data/a.txt"))
        self.assertIsNotNone(reloaded.get_file("data/b.txt"))

    def test_manifest_is_valid_json(self):
        manifest = IndexManifest(self.manifest_path, "idx-1")
        manifest.upsert_file("data/a.txt", {"doc_id": "a", "version": "v1", "chunk_ids": []})
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["index_version"], "idx-1")
        self.assertIn("data/a.txt", data["files"])

    def test_parent_store_replaces_document(self):
        store = ParentStore(self.parent_path, "idx-1")
        store.upsert_document(
            "a",
            {
                "p1": {"parent_id": "p1", "doc_id": "a", "text": "old"},
            },
        )
        store.upsert_document(
            "a",
            {
                "p2": {"parent_id": "p2", "doc_id": "a", "text": "new"},
            },
        )
        self.assertIsNone(store.get("p1"))
        self.assertEqual(store.get("p2")["text"], "new")

        store.remove_document("a")
        self.assertEqual(store.all_parents(), {})


if __name__ == "__main__":
    unittest.main()
