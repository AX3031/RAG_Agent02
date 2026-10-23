import unittest

from langchain_core.documents import Document

from rag.sparse_index import BM25Index, tokenize


class SparseIndexTest(unittest.TestCase):
    def test_tokenize_keeps_cjk_bigrams_and_identifiers(self):
        tokens = tokenize("学而时习之 HD-8842 v2.1")
        self.assertIn("学而", tokens)
        self.assertIn("时习", tokens)
        self.assertIn("hd-8842", tokens)
        self.assertIn("v2.1", tokens)

    def test_bm25_prefers_exact_identifier(self):
        docs = [
            Document(page_content="订单 HD-8842 的状态是已发货", metadata={"chunk_id": "c1"}),
            Document(page_content="退款一般三个工作日到账", metadata={"chunk_id": "c2"}),
            Document(page_content="天气晴朗，适合出行", metadata={"chunk_id": "c3"}),
        ]
        index = BM25Index(docs)
        result = index.search("HD-8842", k=2)
        self.assertEqual(result[0][0].metadata["chunk_id"], "c1")

    def test_allowed_ids_filters_scope(self):
        docs = [
            Document(page_content="学而时习之", metadata={"chunk_id": "c1"}),
            Document(page_content="学而时习之，不亦说乎", metadata={"chunk_id": "c2"}),
        ]
        index = BM25Index(docs)
        result = index.search("学而", k=2, allowed_ids={"c2"})
        self.assertEqual([doc.metadata["chunk_id"] for doc, _ in result], ["c2"])

    def test_empty_query_returns_empty(self):
        index = BM25Index([Document(page_content="内容", metadata={"chunk_id": "c1"})])
        self.assertEqual(index.search("", k=3), [])


if __name__ == "__main__":
    unittest.main()
