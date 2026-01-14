import os
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI

from config.settings import Config  # 假设你把 Config 类放在这里
from core.utils import token_usage_handler

config = Config()
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=config.modelscope_api_key,  # ModelScope Token
)
llm = ChatOpenAI(
    model_name='deepseek-ai/DeepSeek-V3.2',  #
    openai_api_base='https://api-inference.modelscope.cn/v1',
    openai_api_key=config.modelscope_api_key,
    temperature=0,
    streaming=False,
    max_retries=3,
    callbacks=[token_usage_handler]
)