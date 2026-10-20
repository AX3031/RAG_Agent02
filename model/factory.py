from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_openai import ChatOpenAI


import os
from utils.config_handler import rag_conf
from dotenv import load_dotenv
load_dotenv()


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


# class ChatModelFactory(BaseModelFactory):
#     def generator(self) -> Optional[Embeddings | BaseChatModel]:
#         return ChatTongyi(model=rag_conf["chat_model_name"])


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        model_name = rag_conf["chat_model_name"]
        openai_api_key = os.getenv("OPEN_API_KEY")
        openai_api_base = os.getenv("BASE_URL_DASHSCOPE")
        return ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, openai_api_base=openai_api_base)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()

if __name__ == '__main__':
    print(rag_conf["chat_model_name"])