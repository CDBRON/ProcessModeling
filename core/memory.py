import json
import datetime
from typing import List, Dict, Any

class SharedMemory:
    """
    一个用于在多智能体流水线中记录和共享交互历史的模块。
    """
    def __init__(self):
        self.log: List[Dict[str, Any]] = []
        self.costs: List[Dict[str, Any]] = []
        print("--- [Memory] Shared Memory module initialized (with cost tracking). ---")

    def add_entry(self, agent_name: str, input_data: Any, output_data: Any, cost_info: Dict = None):
        """向日志中添加一个新的条目，并可选地记录成本。"""
        timestamp = datetime.datetime.now().isoformat()
        self.log.append({
            "timestamp": timestamp,
            "agent_name": agent_name,
            "input": input_data,
            "output": output_data
        })
        print(f"--- [Memory] Logged entry for {agent_name}. Total entries: {len(self.log)} ---")

        if cost_info:
            self.costs.append({
                "timestamp": timestamp,
                "agent_name": agent_name,
                **cost_info
            })
            print(f"--- [Cost] Logged cost for {agent_name}: "
                  f"Input Tokens: {cost_info.get('input_tokens', 0)}, "
                  f"Output Tokens: {cost_info.get('output_tokens', 0)}")

    def get_formatted_log(self) -> str:
        """将日志格式化为字符串，以便注入到智能体的提示中。"""
        if not self.log:
            return "No interactions have been logged yet."

        formatted_entries = []
        for i, entry in enumerate(self.log):
            # 为了提示的简洁性，对长输出进行截断
            try:
                output_str = json.dumps(entry.get('output', ''), indent=2, ensure_ascii=False)
            except TypeError:
                output_str = str(entry.get('output', ''))

            if len(output_str) > 600:
                output_str = output_str[:600] + "\n... (output truncated)"

            # 同样处理输入
            try:
                input_str = json.dumps(entry.get('input', ''), indent=2, ensure_ascii=False)
            except TypeError:
                input_str = str(entry.get('input', ''))

            if len(input_str) > 400:
                input_str = input_str[:400] + "\n... (input truncated)"


            formatted_entry = (
                f"--- Log Entry {i+1} ---\n"
                f"Agent: {entry['agent_name']}\n"
                f"Timestamp: {entry['timestamp']}\n"
                f"Input:\n{input_str}\n"
                f"Output:\n{output_str}"
            )
            formatted_entries.append(formatted_entry)

        return "\n\n".join(formatted_entries)

    def print_full_log(self):
        """以可读的JSON格式打印完整的日志。"""
        print("\n\n" + "="*40)
        print("====== FULL SHARED MEMORY LOG ======")
        print("="*40 + "\n")
        # 使用 ensure_ascii=False 以正确显示中文字符
        print(json.dumps(self.log, indent=2, ensure_ascii=False))

    def calculate_and_print_total_cost(self):
        """计算并打印详细的总成本报告。"""
        # ModelScope 通义千问系列价格 (截至2024年中期，单位：元/千tokens)
        # 注意：请根据您使用的具体模型和最新价格进行调整
        pricing = {
            'ZhipuAI/GLM-4.5': {'input': 0.0008, 'output': 0.002},
            'ZhipuAI/GLM-4.6': {'input': 0.0008, 'output': 0.002},
            'deepseek-ai/DeepSeek-V3.2-Exp': {'input': 0.0002, 'output': 0.003},
            'deepseek-ai/DeepSeek-V3.1': {'input': 0.0005, 'output': 0.012},
            'MiniMax/MiniMax-M2': {'input': 0.0021, 'output': 0.0084},
            'Qwen/Qwen3-Coder-480B-A35B-Instruct':{'input':0.004,'output':0.016},
            'deepseek-ai/DeepSeek-R1-0528':{'input':0.004,'output':0.016}
        }

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        cost_by_agent = {}

        for cost_entry in self.costs:
            agent_name = cost_entry['agent_name']
            # 从 LangChain 回调中获取的模型名可能与字典中的 key 略有不同，我们需要做一些标准化
            model_name_raw = cost_entry.get('model_name', 'unknown')

            # 【修正后的逻辑】
            # 直接在 pricing 字典里查找模型名称
            model_pricing = pricing.get(model_name_raw)

            input_tokens = cost_entry.get('input_tokens', 0)
            output_tokens = cost_entry.get('output_tokens', 0)

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            cost = 0.0
            if model_pricing:
                cost = (input_tokens / 1000 * model_pricing['input']) + \
                       (output_tokens / 1000 * model_pricing['output'])
                total_cost += cost
            else:
                print(f"!!! WARNING: Pricing for model '{model_name_raw}' not found. Cost for this call will be 0. !!!")

            if agent_name not in cost_by_agent:
                cost_by_agent[agent_name] = {'total_cost': 0.0, 'calls': 0, 'total_input_tokens': 0,
                                             'total_output_tokens': 0}

            cost_by_agent[agent_name]['total_cost'] += cost
            cost_by_agent[agent_name]['calls'] += 1
            cost_by_agent[agent_name]['total_input_tokens'] += input_tokens
            cost_by_agent[agent_name]['total_output_tokens'] += output_tokens

        print("\n\n" + "=" * 40)
        print("====== TOTAL COST REPORT (ACCURATE) ======")
        print("=" * 40 + "\n")
        print(f"Total Input Tokens:  {total_input_tokens}")
        print(f"Total Output Tokens: {total_output_tokens}")
        print(f"Total Estimated Cost: ¥{total_cost:.6f}\n")  # 增加小数位数以显示更精确的成本

        print("--- Cost Breakdown by Agent ---")
        for agent, data in cost_by_agent.items():
            print(f"  - Agent: {agent}")
            print(f"    - Calls: {data['calls']}")
            print(f"    - Total Cost: ¥{data['total_cost']:.6f}")
            print(f"    - Total Input Tokens: {data['total_input_tokens']}")
            print(f"    - Total Output Tokens: {data['total_output_tokens']}")

        print("\n" + "=" * 40)