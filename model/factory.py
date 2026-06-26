"""
使用抽象工厂模式创建和管理聊天模型和嵌入模型的实例，支持阿里云通义千问系列模型
"""
# 导入抽象基类，用于定义工厂接口
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf




#定义抽象模型工厂基类
class BaseModelFactory(ABC):
    # 定义抽象方法，子类必须实现该方法
    @abstractmethod
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        #抽象方法不包含实现逻辑，用于定义接口规范
        pass






#聊天模型工厂类，继承抽象工厂
class ChatModelFactory(BaseModelFactory):
    #实现抽象方法，创建聊天模型实例
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        #创建并返回通义千问聊天模型实例，使用配置中的模型名称
        # 直接在此处填写你的 DASHSCOPE_API_KEY
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            api_key="sk-b371ebde92284d9ebf00b32645ea6edd"
        )





#嵌入模型工厂类，继承抽象工厂
class EmbeddingModelFactory(BaseModelFactory):
    #实现抽象方法，创建嵌入模型实例
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])





#在模块导入时立即创建模型实例，拱全局使用
chat_model = ChatModelFactory().generator()                     #全局聊天模型实例
embed_model = EmbeddingModelFactory().generator()           #全局嵌入模型实例





















