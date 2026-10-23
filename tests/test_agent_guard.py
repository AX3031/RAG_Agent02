import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.react_agent import _knowledge_insufficient


class AgentGuardTest(unittest.TestCase):
    def _tool(self, name: str, payload: dict) -> ToolMessage:
        return ToolMessage(content=json.dumps(payload, ensure_ascii=False), name=name, tool_call_id="1")

    def test_no_evidence_blocks_hard_generation(self):
        messages = [
            HumanMessage(content="问题"),
            self._tool("retrieve_knowledge", {"status": "no_evidence", "answerable": False}),
            AIMessage(content="模型自由发挥"),
        ]
        self.assertTrue(_knowledge_insufficient(messages))

    def test_error_blocks_hard_generation(self):
        messages = [self._tool("retrieve_knowledge", {"status": "error", "answerable": False})]
        self.assertTrue(_knowledge_insufficient(messages))

    def test_answerable_allows_generation(self):
        messages = [self._tool("retrieve_knowledge", {"status": "ok", "answerable": True})]
        self.assertFalse(_knowledge_insufficient(messages))

    def test_web_search_overrides_refusal(self):
        messages = [
            self._tool("retrieve_knowledge", {"status": "no_evidence", "answerable": False}),
            self._tool("web_search_baidu", {"results": "实时信息"}),
        ]
        self.assertFalse(_knowledge_insufficient(messages))

    def test_bad_json_is_not_blocking(self):
        messages = [ToolMessage(content="not-json", name="retrieve_knowledge", tool_call_id="1")]
        self.assertFalse(_knowledge_insufficient(messages))


if __name__ == "__main__":
    unittest.main()
