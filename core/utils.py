import sys
import tiktoken
from typing import Dict, List, Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class TokenUsageCallbackHandler(BaseCallbackHandler):
    
    def __init__(self, model_encoding="cl100k_base"):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
        try:
            self.encoder = tiktoken.get_encoding(model_encoding)
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        
        self._current_input_tokens = 0

    def _count_tokens(self, text: str) -> int:
        
        if not text:
            return 0
        try:
            return len(self.encoder.encode(text))
        except Exception:
           
            return int(len(text) * 0.7)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
       
        count = 0
        for p in prompts:
            count += self._count_tokens(p)

        self._current_input_tokens = count
        

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        
        usage = None
        if response.generations:
            generation = response.generations[0][0]
            if hasattr(generation, 'message') and hasattr(generation.message, 'response_metadata'):
                usage = generation.message.response_metadata.get('token_usage') or generation.message.response_metadata.get('usage')
            elif hasattr(generation, 'generation_info'):
                usage = generation.generation_info.get('token_usage') or generation.generation_info.get('usage')

        if not usage and response.llm_output:
            usage = response.llm_output.get('token_usage') or response.llm_output.get('usage')

      
        if usage:
            
            p_tokens = usage.get('prompt_tokens') or usage.get('input_tokens') or 0
            c_tokens = usage.get('completion_tokens') or usage.get('output_tokens') or 0
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens
            
        else:
            
            p_tokens = self._current_input_tokens

            
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
   
    token_usage_handler.get_and_reset_totals()

    print(f"--- [Cost Logging] Invoking {agent_name}... ---")

    
    response = agent_executor.invoke(agent_input)

   
    usage_totals = token_usage_handler.get_and_reset_totals()
    input_tokens = usage_totals["prompt_tokens"]
    output_tokens = usage_totals["completion_tokens"]

    
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
