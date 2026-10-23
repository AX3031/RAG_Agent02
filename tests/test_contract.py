import unittest

from rag.contract import ErrorCode, RetrievalResult, Status, validate_evidence_ids


class ContractTest(unittest.TestCase):
    def test_error_result_contract(self):
        result = RetrievalResult.error_result(
            request_id="req_1",
            trace_id="trace_1",
            index_version="idx-1",
            code=ErrorCode.INVALID_QUERY,
            message="empty",
        )
        data = result.to_dict()
        self.assertEqual(data["status"], "error")
        self.assertFalse(data["answerable"])
        self.assertEqual(data["error"]["code"], "INVALID_QUERY")
        self.assertEqual(data["contract_version"], "1.0")
        self.assertEqual(data["request_id"], "req_1")
        self.assertEqual(data["trace_id"], "trace_1")
        self.assertEqual(data["index_version"], "idx-1")

    def test_validate_evidence_ids(self):
        allowed = ["ev_001", "ev_002"]
        self.assertEqual(validate_evidence_ids("结论见【ev_001】", allowed), [])
        self.assertEqual(validate_evidence_ids("结论见【参考资料2】", allowed), [])
        self.assertEqual(len(validate_evidence_ids("编造的【ev_009】", allowed)), 1)
        self.assertEqual(len(validate_evidence_ids("编造的【参考资料9】", allowed)), 1)

    def test_status_enum_matches_v2(self):
        self.assertEqual(
            {item.value for item in Status},
            {"ok", "no_evidence", "degraded", "denied", "error"},
        )


if __name__ == "__main__":
    unittest.main()
