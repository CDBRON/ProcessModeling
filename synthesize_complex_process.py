import os
import json
import random
import uuid
import time
import networkx as nx
from typing import List, Dict, Optional
from openai import OpenAI


import config

config = config.Config()


MODEL_NAME = ""
INPUT_POOL_DIR = "./auto_pool"  
OUTPUT_DATASET_DIR = "./synthetic_complex_dataset"


# ========================================

class ProcessSynthesizer:
    def __init__(self):
        self.client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key=config.modelscope_api_key,
        )
        self.pool = self._load_pool()

    def _load_pool(self) -> Dict[str, List[Dict]]:
      
        pool = {}
        if not os.path.exists(INPUT_POOL_DIR):
            print(f"错误：输入目录 {INPUT_POOL_DIR} 不存在！")
            return {}

        files = [f for f in os.listdir(INPUT_POOL_DIR) if f.endswith('.json')]
        print(f"正在加载 {len(files)} 个原子流程...")

        for f in files:
            try:
                with open(os.path.join(INPUT_POOL_DIR, f), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    domain = data.get("metadata", {}).get("domain", "Uncategorized")
                    if domain not in pool:
                        pool[domain] = []
                    pool[domain].append(data)
            except Exception as e:
                print(f"加载文件 {f} 失败: {e}")

        print(f"加载完成。领域分布: { {k: len(v) for k, v in pool.items()} }")
        return pool

    def _select_compatible_atoms(self, domain: str, num_nodes: int) -> List[Dict]:
      
        candidates = self.pool.get(domain, [])
        if len(candidates) < num_nodes:
            print(f"警告：领域 {domain} 的样本不足（只有 {len(candidates)} 个，需要 {num_nodes} 个）")
            return []
        return random.sample(candidates, num_nodes)

    def _generate_bridge_logic(self, prev_atom: Dict, next_atom: Dict) -> str:
      
        prompt = f"""
        You are a Business Process Expert.
        Task: Write a transition sentence to connect Process A and Process B logically.

        Process A: "{prev_atom['original_text']}"
        Process B: "{next_atom['original_text']}"

        Constraint: The transition should explain how the outcome of A triggers B. Keep it concise (1-2 sentences).
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成桥接逻辑失败: {e}")
            return "Then, the process continues to the next stage."

    def _generate_coarse_instruction(self, fine_grained_text: str) -> str:
       
        prompt = f"""
        You are an expert Business Process Architect. Your task is to read a long, detailed business process description and summarize it into a single, concise, one-sentence "design brief".

        This "design brief" is a command that will be given to another AI agent to reconstruct the process. Therefore, it must contain all the critical "golden triangle" constraints: Roles, Key Activities, and Key Decision Points.
    
        **YOUR THINKING PROCESS:**
        1.  **Identify All Roles**: First, read through the entire text and list every unique role involved (e.g., "customer", "system", "manager").
        2.  **Identify Key Activities**: Next, identify the most critical, value-adding activities. Ignore minor, supporting, or technical steps. Ask yourself, "If I remove this activity, does the process fundamentally break?" If yes, it's a key activity. Aim for 2-4 key activities.
        3.  **Identify Key Decision Points (Gateways)**: Then, find the most important questions or branching points in the process. These are usually explicitly stated as "if/then" or decision points. Aim for 2-3 key decision points.
        4.  **Synthesize into One Sentence**: Finally, combine all the identified elements into a single, fluent, natural language sentence following the required format.
    
        **OUTPUT FORMAT:**
        Your output MUST be a single sentence following this template:
        "Please design a [Process Name] that includes the roles of [Role 1] and [Role 2]. The key activities are [Key Activity 1] and [Key Activity 2], and there should be decision points regarding [Key Decision 1] and [Key Decision 2]."
    
        **CRITICAL RULES:**
        *   The output must be **one single sentence**.
        *   Use single quotes for roles, activities, and decision points.
        *   List roles, activities, and decision points clearly. Do not merge them.
        *   The language should be natural and instructive, as if you are commanding another agent.
    
        ---
        **EXAMPLE 1:**
    
        **Input Text:**
        "The process begins when a customer submits a credit application. The system first checks if the application is complete. If not, it notifies the customer. If complete, the credit company performs a risk assessment. Based on the risk level and the requested amount, a decision is made. If the amount is high, a special approval must be requested. If the risk is high, the application is rejected. Finally, the customer is notified of the approval or rejection."
    
        **Your Correct Output:**
        Please design a credit approval process that includes the roles of 'customer' and 'credit company'. The key activities are 'risk assessment' and 'request for approval', and there should be decision points regarding 'request amount' and 'risk assessment'.
    
        ---
        **EXAMPLE 2:**
    
        **Input Text:**
        "Our new employee onboarding starts with the HR department sending an offer. Once the candidate accepts, the IT department and the Facilities department work in parallel. IT sets up the laptop and accounts, while Facilities prepares the desk. The new employee then meets their manager for an orientation. After that, the employee can choose to enroll in health insurance, a retirement plan, or both."
    
        **Your Correct Output:**
        Please design an employee onboarding process that includes the roles of 'HR department', 'candidate', 'IT department', and 'Facilities department'. The key activities are 'send offer' and 'manager orientation', and there should be decision points regarding 'offer acceptance' and 'benefits enrollment'.
    
        ---
    
        **YOUR CURRENT TASK:**
        Now, apply this exact logic to the following detailed process description.
        """

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content":prompt},
                    {"role": "user", "content": f'Now, apply this exact logic to the following detailed process description.**Input Text:**{fine_grained_text}'}
                ],
                temperature=0.1 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成粗粒度指令失败: {e}")
            return "Error generating instruction."

    def synthesize_one_sample(self) -> Optional[Dict]:
        
        valid_domains = [d for d, items in self.pool.items() if len(items) >= 3]
        if not valid_domains:
            print("没有足够数据的领域（至少需要3个原子流程）！")
            return None

        domain = random.choice(valid_domains)
        print(f"--- 选中领域: {domain} ---")

      
        chain_length = random.randint(7, 7)
        print(f"--- 计划合成长度: {chain_length} 个原子流程 ---")

        atoms = self._select_compatible_atoms(domain, chain_length)
        if not atoms: return None

       
        full_text_parts = []
        bridge_logics = []

        print("--- 正在生成桥接逻辑... ---")
        for i in range(len(atoms)):
            
            full_text_parts.append(f"Step {i + 1}: {atoms[i]['original_text']}")

            
            if i < len(atoms) - 1:
                bridge = self._generate_bridge_logic(atoms[i], atoms[i + 1])
                full_text_parts.append(f"[Transition]: {bridge}")
                bridge_logics.append(bridge)
                print(f"  > Bridge {i}->{i + 1} generated.")

        fine_grained_text = "\n".join(full_text_parts)
        print(f"--- 细粒度文本合成完毕 (长度: {len(fine_grained_text)} 字符) ---")

        
        print("--- 正在逆向生成粗粒度指令... ---")
        coarse_request = self._generate_coarse_instruction(fine_grained_text)
        print(f"--- 粗粒度指令: {coarse_request} ---")

        return {
            "id": str(uuid.uuid4()),
            "domain": domain,
            "complexity_level": chain_length,
            "coarse_request": coarse_request,
            "fine_grained_text": fine_grained_text,
            "atomic_components": [a.get('id', 'unknown') for a in atoms],
            "bridge_logics": bridge_logics,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def generate_single_sample(self):
       
        if not os.path.exists(OUTPUT_DATASET_DIR):
            os.makedirs(OUTPUT_DATASET_DIR)

        print("\n=== 开始生成单条复杂流程数据 ===")
        sample = self.synthesize_one_sample()

        if sample:
           
            file_name = f"complex_{sample['id']}.json"
            file_path = os.path.join(OUTPUT_DATASET_DIR, file_name)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)

            print(f"\n✅ 成功生成数据！已保存至: {file_path}")
            print(f"    - Domain: {sample['domain']}")
            print(f"    - Complexity: {sample['complexity_level']}")
            return file_path
        else:
            print("\n❌ 生成失败。")
            return None


if __name__ == "__main__":
    synthesizer = ProcessSynthesizer()
 

    synthesizer.generate_single_sample()
