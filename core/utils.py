import sys
import tiktoken
from typing import Dict, List, Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class TokenUsageCallbackHandler(BaseCallbackHandler):
    """
    本地估算版 Token 处理器。
    如果 API 不返回 usage 信息，则使用 tiktoken 在本地计算。
    """
    def __init__(self, model_encoding="cl100k_base"):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # 加载分词器，cl100k_base 是 GPT-4/3.5 的标准，对大多数模型估算足够准确
        try:
            self.encoder = tiktoken.get_encoding(model_encoding)
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        # 临时存储本次调用的 input token，因为 on_llm_end 拿不到 prompt 文本
        self._current_input_tokens = 0

    def _count_tokens(self, text: str) -> int:
        """本地计算字符串的 token 数"""
        if not text:
            return 0
        try:
            return len(self.encoder.encode(text))
        except Exception:
            # 降级处理：如果编码失败，按字符数粗略估算 (中文约0.7个token/字)
            return int(len(text) * 0.7)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """在 LLM 开始时触发，用于计算 Input Tokens"""
        count = 0
        for p in prompts:
            count += self._count_tokens(p)

        self._current_input_tokens = count
        # print(f"   [Local Calc] Input text length: {len(prompts[0])}, Est. Tokens: {count}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """在 LLM 结束时触发，优先读 API，读不到则本地计算 Output"""

        # 1. 尝试从 API 元数据获取 (你之前的逻辑)
        usage = None
        if response.generations:
            generation = response.generations[0][0]
            if hasattr(generation, 'message') and hasattr(generation.message, 'response_metadata'):
                usage = generation.message.response_metadata.get('token_usage') or generation.message.response_metadata.get('usage')
            elif hasattr(generation, 'generation_info'):
                usage = generation.generation_info.get('token_usage') or generation.generation_info.get('usage')

        if not usage and response.llm_output:
            usage = response.llm_output.get('token_usage') or response.llm_output.get('usage')

        # 2. 分支处理
        if usage:
            # A. 如果 API 返回了数据，直接用 API 的
            p_tokens = usage.get('prompt_tokens') or usage.get('input_tokens') or 0
            c_tokens = usage.get('completion_tokens') or usage.get('output_tokens') or 0
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens
            # print(f"   [Source: API] In: {p_tokens}, Out: {c_tokens}")
        else:
            # B. 如果 API 没返回 (你当前的情况)，使用本地计算
            # Input: 使用 on_llm_start 算好的
            p_tokens = self._current_input_tokens

            # Output: 计算生成内容的 token
            c_tokens = 0
            if response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        c_tokens += self._count_tokens(gen.text)

            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens
            print(f"   [Source: Local Calc] In: {p_tokens}, Out: {c_tokens}")

    def get_and_reset_totals(self) -> Dict[str, int]:
        totals = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._current_input_tokens = 0
        return totals

class TeeOutput:
    def __init__(self, filename):
        self.filename = filename
        self.original_stdout = sys.stdout
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, 'w', encoding='utf-8')
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()

    def write(self, message):
        self.original_stdout.write(message)
        if self.file:
            self.file.write(message)

    def flush(self):
        self.original_stdout.flush()
        if self.file:
            self.file.flush()


def invoke_with_cost_logging(agent_executor, agent_input, agent_name, memory_module,llm_object):
    """
    包装 AgentExecutor 的 invoke 方法，使用自定义 Handler 捕获 token。
    """
    # 1. 先清零（防止之前的残留）
    token_usage_handler.get_and_reset_totals()

    print(f"--- [Cost Logging] Invoking {agent_name}... ---")

    # 2. 执行 Agent
    # 注意：因为我们在 LLM 初始化时已经绑定了 handler，这里不需要在 invoke 里再传 callbacks
    response = agent_executor.invoke(agent_input)

    # 3. 执行后获取累计值
    usage_totals = token_usage_handler.get_and_reset_totals()
    input_tokens = usage_totals["prompt_tokens"]
    output_tokens = usage_totals["completion_tokens"]

    # 4. 打印调试信息，确认是否捕获到
    if input_tokens == 0 and output_tokens == 0:
        print(f"!!! [Cost Error] {agent_name} returned 0 tokens. Check the '[Token Warning]' logs above.")
    else:
        print(f"--- [Cost Success] {agent_name}: In={input_tokens}, Out={output_tokens}")

    cost_info = {
        "model_name": llm_object.model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": 0
    }

    memory_module.add_entry(agent_name, agent_input, response, cost_info)
    return response

token_usage_handler = TokenUsageCallbackHandler()