import hashlib
import os
import tempfile
import unittest

from langchain_core.documents import Document

from rag.chunking import compute_index_version, split_documents, stable_doc_id


class ChunkingTest(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".txt", text=True)
        os.close(fd)
        content = "# 第一章\n第一条内容，讲学习。\n第二条内容，讲实践。\n\n## 1.1 小节\n小节里的内容。\n"
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(content)
        self.documents = [Document(page_content=content, metadata={"page": 0})]

    def tearDown(self):
        os.unlink(self.path)

    def _split(self):
        return split_documents(
            source_path=self.path,
            rel_path="data/test.txt",
            documents=self.documents,
            index_version="idx-test",
            embedding_model="fake-embedding",
            chunk_size=80,
            chunk_overlap=10,
            parent_size=200,
            separators=["\n\n", "\n", "。", " ", ""],
        )

    def test_doc_id_is_stable(self):
        self.assertEqual(stable_doc_id("data/论语.pdf"), stable_doc_id("data/论语.pdf"))
        self.assertNotEqual(stable_doc_id("data/a.pdf"), stable_doc_id("data/b.pdf"))

    def test_index_version_changes_with_embedding(self):
        self.assertNotEqual(
            compute_index_version("m1", 200, 20, 800),
            compute_index_version("m2", 200, 20, 800),
        )

    def test_children_have_required_metadata(self):
        bundle = self._split()
        self.assertGreater(len(bundle.child_docs), 0)
        self.assertGreater(len(bundle.parents), 0)
        for doc in bundle.child_docs:
            for key in (
                "doc_id",
                "version",
                "chunk_id",
                "parent_id",
                "seq",
                "source",
                "page",
                "char_start",
                "char_end",
                "title_path",
                "content_hash",
                "scope_tags",
                "embedding_model",
                "index_version",
                "is_active",
            ):
                self.assertIn(key, doc.metadata)
            self.assertIn(doc.metadata["parent_id"], bundle.parents)
            self.assertEqual(doc.metadata["index_version"], "idx-test")

    def test_ids_are_deterministic(self):
        first = self._split()
        second = self._split()
        self.assertEqual(
            [doc.metadata["chunk_id"] for doc in first.child_docs],
            [doc.metadata["chunk_id"] for doc in second.child_docs],
        )

    def test_heading_is_included_in_text_and_title_path(self):
        bundle = self._split()
        joined = "\n".join(doc.page_content for doc in bundle.child_docs)
        self.assertIn("第一章", joined)
        self.assertTrue(any("第一章" in str(doc.metadata["title_path"]) for doc in bundle.child_docs))

    def test_content_hash_matches_text(self):
        bundle = self._split()
        for doc in bundle.child_docs:
            expected = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
            self.assertEqual(doc.metadata["content_hash"], expected)


if __name__ == "__main__":
    unittest.main()
