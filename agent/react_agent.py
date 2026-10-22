import os

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id, get_current_month,
                                     fetch_external_data, fill_context_for_report, web_search_baidu)
from agent.tools.middleware import (monitor_tool, log_before_model, report_prompt_switch,
                                    pii_middleware_email, pii_middleware_phone, model_fallback_middleware)


def _build_checkpointer():
    """仅当显式开启时才连接 PostgreSQL，避免默认启动依赖数据库。"""
    if os.getenv("USE_POSTGRES_CHECKPOINTER", "").lower() in {"1", "true", "yes", "on"}:
        # 延迟导入：未开启时即使没装 psycopg 也不影响启动
        from memorys.postgres_checkpointer import get_checkpointer
        return get_checkpointer()
    return None


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id, get_current_month,
                   fetch_external_data, fill_context_for_report, web_search_baidu],
            middleware=[model_fallback_middleware, pii_middleware_email, pii_middleware_phone,
                        monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=_build_checkpointer(),
        )

    def execute_stream(self, query: str, thread_id: str | None = None):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }

        # thread_id 用于会话状态持久化；未开启 checkpointer 时传 None 即可
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None

        # 第三个参数context就是上下文runtime中的信息，就是我们提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}, config=config):
            lastest_message = chunk["messages"][-1]
            if lastest_message.content:
                yield lastest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("我这里天气怎么样", thread_id="local_test"):
        print(chunk, end="", flush=True)