import math
import unittest

from eval.retrieval_metrics import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    normalize,
    random_recall_at_k,
    recall_at_k,
    resolve_snippets,
)


class _Doc:
    def __init__(self, chunk_id: str, text: str):
        self.page_content = text
        self.metadata = {"chunk_id": chunk_id}


class MetricsTest(unittest.TestCase):
    def test_normalize_removes_punctuation_and_space(self):
        self.assertEqual(normalize("子曰： “ 学而 时习之 ”"), "子曰学而时习之")
        self.assertEqual(normalize("HD-8842 / v2.1"), "hd8842v21")

    def test_recall(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"b", "d"}, 3), 0.5)
        self.assertEqual(recall_at_k(["a"], {"b"}, 3), 0.0)

    def test_hit(self):
        self.assertEqual(hit_at_k(["a", "b"], {"b"}, 2), 1.0)
        self.assertEqual(hit_at_k(["a"], {"b"}, 2), 0.0)

    def test_mrr(self):
        self.assertAlmostEqual(mrr_at_k(["a", "b", "c"], {"b"}, 3), 0.5)
        self.assertAlmostEqual(mrr_at_k(["a", "b"], {"b"}, 2), 0.5)
        self.assertEqual(mrr_at_k(["a"], {"b"}, 2), 0.0)

    def test_ndcg_perfect_order(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a", "b"}, 2), 1.0)
        self.assertAlmostEqual(ndcg_at_k(["b", "a"], {"a", "b"}, 2), 1.0)
        self.assertLess(ndcg_at_k(["x", "a"], {"a"}, 2), 1.0)
        self.assertAlmostEqual(ndcg_at_k(["x", "a"], {"a"}, 2), 1 / math.log2(3))

    def test_random_recall_bounds(self):
        self.assertEqual(random_recall_at_k(10, 0, 10), 0.0)
        self.assertEqual(random_recall_at_k(5, 3, 10), 1.0)
        value = random_recall_at_k(100, 5, 10)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 0.6)

    def test_resolve_snippets(self):
        docs = [
            _Doc("c1", "子曰：学而时习之，不亦说乎？"),
            _Doc("c2", "有朋自远方来，不亦乐乎？"),
            _Doc("c3", "今天天气很好"),
        ]
        resolved, missing = resolve_snippets(["学而时习之"], docs)
        self.assertEqual(resolved, {"c1"})
        self.assertEqual(missing, [])

        # 跨块片段：按标点切分后分别定位
        resolved, missing = resolve_snippets(["学而时习之，有朋自远方来"], docs)
        self.assertEqual(resolved, {"c1", "c2"})
        self.assertEqual(missing, [])

        resolved, missing = resolve_snippets(["不存在的句子"], docs)
        self.assertEqual(resolved, set())
        self.assertEqual(missing, ["不存在的句子"])

    def test_resolve_single_greek_identifier(self):
        docs = [_Doc("c1", "judge 与人工标注计算 Cohen's κ；κ < 0.6")]
        resolved, missing = resolve_snippets(["κ"], docs)
        self.assertEqual(resolved, {"c1"})
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
