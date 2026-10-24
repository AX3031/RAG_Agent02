import unittest

from langchain_core.documents import Document

from rag.retrieval_service import RetrievalService
from rag.telemetry import TTLCache
from rag.vector_store import VectorStoreService


def _doc(chunk_id: str) -> Document:
    return Document(page_content=chunk_id, metadata={"chunk_id": chunk_id})


class FusionTest(unittest.TestCase):
    def setUp(self):
        self.service = object.__new__(RetrievalService)
        self.service.cfg = {"rrf_k": 60}

    def test_rrf_prefers_document_hit_by_both_paths(self):
        fused = self.service._rrf_fuse(
            [(_doc("a"), 0.9), (_doc("b"), 0.8)],
            [(_doc("b"), 12.0), (_doc("c"), 5.0)],
        )
        self.assertEqual(fused[0]["chunk_id"], "b")
        self.assertGreater(fused[0]["rrf"], fused[1]["rrf"])

    def test_rrf_keeps_all_candidates(self):
        fused = self.service._rrf_fuse([(_doc("a"), 0.9)], [(_doc("b"), 1.0)])
        self.assertEqual({item["chunk_id"] for item in fused}, {"a", "b"})

    def test_rrf_scores_are_sorted(self):
        fused = self.service._rrf_fuse(
            [(_doc("a"), 0.9), (_doc("b"), 0.8), (_doc("c"), 0.7)],
            [(_doc("c"), 9.0)],
        )
        scores = [item["rrf"] for item in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_build_where(self):
        self.assertIsNone(VectorStoreService._build_where(None))
        self.assertEqual(VectorStoreService._build_where({"doc_id": "a"}), {"doc_id": "a"})
        self.assertEqual(
            VectorStoreService._build_where({"doc_id": "a", "scope_tags": ["public"]}),
            {"$and": [{"doc_id": "a"}, {"scope_tags": '["public"]'}]},
        )

    def test_is_included_filters_by_filename_keyword(self):
        service = object.__new__(VectorStoreService)
        service.include_keywords = ["道德经"]
        service.exclude_keywords = []
        self.assertTrue(service._is_included("data/《道德经》带讲解.pdf"))
        self.assertFalse(service._is_included("data/论语.pdf"))

        service.include_keywords = []
        service.exclude_keywords = ["道德经"]
        self.assertFalse(service._is_included("data/《道德经》带讲解.pdf"))
        self.assertTrue(service._is_included("data/论语.pdf"))


class CacheTest(unittest.TestCase):
    def test_set_get_and_evict(self):
        cache = TTLCache(max_size=2, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)

    def test_ttl_expiry(self):
        cache = TTLCache(max_size=2, ttl_seconds=0.05)
        cache.set("a", 1)
        import time

        time.sleep(0.08)
        self.assertIsNone(cache.get("a"))


if __name__ == "__main__":
    unittest.main()
