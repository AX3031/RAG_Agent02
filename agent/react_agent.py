import json
import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from memorys.memory_factory import DEFAULT_MEMORY_MODE, create_checkpointer
from agent.tools.agent_tools import (retrieve_knowledge, rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report,
                                     web_search_baidu)
from agent.tools.middleware import (monitor_tool, log_before_model, report_prompt_switch,
                                    pii_middleware_email, pii_middleware_phone, model_fallback_middleware)


REFUSAL_TEXT = "根据现有资料无法回答该问题。"


def _knowledge_insufficient(messages: list) -> bool:
    """结构化防御：最后一次知识库检索明确证据不足时，禁止主 Agent 用自有知识硬编。"""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        name = getattr(message, "name", "") or ""
        if name in {"web_search_baidu", "fetch_external_data"}:
            return False
        if name != "retrieve_knowledge":
            continue
        try:
            payload = json.loads(message.content) if isinstance(message.content, str) else message.content
        except (json.JSONDecodeError, TypeError):
            return False
        return bool(payload.get("status") in {"no_evidence", "error"} or not payload.get("answerable"))
    return False


def _default_memory_mode() -> str:
    """兼容旧配置：未显式指定记忆方式时，环境变量仍可切到 PostgreSQL。"""
    if os.getenv("USE_POSTGRES_CHECKPOINTER", "").lower() in {"1", "true", "yes", "on"}:
        return "postgres"
    return DEFAULT_MEMORY_MODE


class ReactAgent:
    def __init__(self, memory_mode: str | None = None):
        # 记忆方式：short_term（默认）/ file / postgres
        self.memory_mode = memory_mode or _default_memory_mode()
        self.checkpointer = create_checkpointer(self.memory_mode)
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[retrieve_knowledge, rag_summarize, get_weather, get_user_location, get_user_id, get_current_month,
                   fetch_external_data, fill_context_for_report, web_search_baidu],
            middleware=[model_fallback_middleware, pii_middleware_email, pii_middleware_phone,
                        monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=self.checkpointer,
        )

    def execute_stream(self, query: str, thread_id: str | None = None):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        # checkpointer 始终存在，thread_id 用于区分会话（相同 ID 恢复历史）
        config = {"configurable": {"thread_id": thread_id or "default"}}

        # 第三个参数context就是上下文runtime中的信息，就是我们提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}, config=config):
            messages = chunk["messages"]
            lastest_message = messages[-1]
            # 只输出模型回答，避免把用户问题、工具结果回显到页面
            if isinstance(lastest_message, AIMessage) and not lastest_message.tool_calls:
                if isinstance(lastest_message.content, str) and lastest_message.content.strip():
                    content = lastest_message.content.strip()
                    if _knowledge_insufficient(messages):
                        content = REFUSAL_TEXT
                    yield content + "\n"

    def clear_memory(self, thread_id: str) -> None:
        """删除指定会话在后端 checkpointer 中的全部状态。"""
        if thread_id and self.checkpointer is not None:
            self.checkpointer.delete_thread(thread_id)

    def load_ui_messages(self, thread_id: str) -> list[dict]:
        """读取指定会话的历史消息，供 UI 在刷新/切换后恢复聊天记录。"""
        if not thread_id or self.checkpointer is None:
            return []

        checkpoint_tuple = self.checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        if checkpoint_tuple is None:
            return []

        messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
        ui_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage) and not message.tool_calls:
                role = "assistant"
            else:
                continue
            if isinstance(message.content, str) and message.content.strip():
                ui_messages.append({"role": role, "content": message.content.strip()})
        return ui_messages


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("我这里天气怎么样", thread_id="local_test"):
        print(chunk, end="", flush=True)