import json
import re
from typing import Union, List, Dict, Any, cast
from dataclasses import dataclass, field

# --- LangChain / LangGraph 导入 ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# --- 项目内部导入 (假设你已经按之前的建议建立了 core 和 tools 目录) ---
from core.llm import llm, client
from tools.classifier import activity_classifier_pipeline # 从 tools 中导入加载好的模型

@dataclass
class GraphState:
    input_json: Dict = field(default_factory=dict)
    processed_json: Dict = field(default_factory=dict)
    activities_to_refine: List[Dict] = field(default_factory=list)
    iteration_count: int = 0


def critique_node(state: GraphState) -> dict:
    print("\n--- [ALIGNER_NODE] Critiquing Granularity ---")
    current_json = state.processed_json
    activities = current_json.get("activities", [])
    activities_to_refine = []
    for activity in activities:
        description = activity.get("description", "")

        # --- 修改开始：提取动作文本，去除角色前缀 ---
        # 默认使用完整描述
        action_text_for_eval = description
        # 如果包含冒号，则分割并取后半部分（即动作部分）
        if ":" in description:
            # split(":", 1) 确保只分割第一个冒号，防止动作内容里也有冒号被误切
            parts = description.split(":", 1)
            if len(parts) > 1:
                action_text_for_eval = parts[1].strip()
        # --- 修改结束 ---

        label = "Standard"
        if activity_classifier_pipeline:
            # 这里传入处理过的 action_text_for_eval 而不是原始 description
            result = activity_classifier_pipeline(action_text_for_eval)
            label = result[0]['label']
        if label != "Standard":
            # 打印日志时同时显示原始描述和用于评估的文本，方便调试
            print(f"  - Found non-standard activity: '{description}' (Evaluated: '{action_text_for_eval}') -> {label}")
            activities_to_refine.append({"activity": activity, "granularity": label})
    return {"activities_to_refine": activities_to_refine}


def refine_node(state: GraphState) -> dict:
    print("\n--- [ALIGNER_NODE] Refining Activities (1-to-1 Standardization) ---")
    current_json = state.processed_json
    activities_to_refine = state.activities_to_refine
    all_activities = current_json.get("activities", [])

    # 提取描述用于集合运算
    problematic_descriptions = {item['activity']['description'] for item in activities_to_refine}

    # 分离标准活动和问题活动
    standard_activities = [act for act in all_activities if act['description'] not in problematic_descriptions]
    problematic_activities = [item['activity'] for item in activities_to_refine]

    print(f"  - Found {len(standard_activities)} standard activities (Keep as is).")
    print(f"  - Found {len(problematic_activities)} problematic activities (To be renamed/standardized).")

    if not problematic_activities:
        print("  - No problematic activities found. Skipping refinement.")
        return {
            "processed_json": current_json,
            "iteration_count": state.iteration_count + 1
        }

    # 生成详细的错误报告，包含具体的粒度问题 (Too Fine / Too Coarse)
    # 格式: "- [Too Fine] 'client: Access website'"
    error_report_lines = []
    for item in activities_to_refine:
        issue_type = item['granularity']  # "Too Fine" or "Too Coarse"
        desc = item['activity']['description']
        error_report_lines.append(f"- [{issue_type}] Activity: '{desc}'")
    error_report_str = "\n".join(error_report_lines)

    standard_activity_examples = """
        "Receive customer inquiry",
        "Address customer concerns",
        "Collect customer information",
        "Provide quote",
        "Place order",
        "Record order in system",
        "Send order confirmation",
        "Conduct initial phone interviews",
        "Check references",
        "Extend offer",
        "Negotiate salary",
        "Check current inventory level",
        "Place order with suppliers",
        "Receive stock",
        "Inspect stock for quality",
        "Conduct site visit",
        "Select supplier",
        "Begin contract negotiations",
        "Sign contract",
        "Onboard supplier",
        "Execute contract",
        "Propose corrective actions",
        "Implement fix",
        "Change policy",
        "Conduct training",
        "Conduct follow-up",
        "Close incident report",
        "Notify all stakeholders",
        "Identify idea for new product or improvement",
        """.strip()

    # --- 核心修改：Prompt 逻辑重构 ---
    refinement_prompt = f"""
        You are an expert BPMN Process Refiner. Your goal is to standardize the descriptions of specific activities to satisfy a strict granularity classifier.

        ### THE STRATEGY: 1-TO-1 REPLACEMENT
        **DO NOT SPLIT activities.** (Splitting causes explosion).
        **DO NOT MERGE activities.** (Merging causes confusion).

        **Your Task:** Take each problematic activity and **RENAME / REPHRASE** it into exactly **ONE** "Standard" operational activity.

        ### INSTRUCTIONS BY TYPE

        1. **IF FLAGGED "TOO COARSE" (Too Vague/Abstract):**
           - **Problem:** The description sounds like a phase or a goal (e.g., "Manage Recruitment").
           - **Fix:** Rename it to the *primary concrete action* represented by that step.
           - **Example:** "HR: Manage Recruitment" -> "HR: Execute recruitment workflow".
           - **Example:** "Department: Define Responsibilities" -> "Department: Document core role duties".

        2. **IF FLAGGED "TOO FINE" (Too Detailed/Mechanical):**
           - **Problem:** The description sounds like a keystroke or micro-step (e.g., "Click Submit", "Type Name").
           - **Fix:** Rename it to the *business-level task* it represents.
           - **Example:** "Candidate: Type Name" -> "Candidate: Enter personal details".
           - **Example:** "System: Click Save" -> "System: Record data entry".

        3. **THE "JUDGE" RULE (Crucial):**
           - The automated classifier is overly sensitive. 
           - If an activity looks reasonable (e.g., "Schedule interview"), **DO NOT CHANGE IT**. Just output it exactly as it was.
        4. **ANALYZE THE ERROR REPORT**: Look at the list below to see WHY each activity is wrong.
        5. **PRESERVE CONTEXT**: Keep the "Role: Action" format (e.g., "client: Submit form").
        6. **INTEGRATE**: Combine your *newly created* activities with the *unchanged* standard activities.
        ---
        ### PART 1: ERROR REPORT (TARGETS FOR MODIFICATION)
        **Do NOT output these descriptions as they are. You MUST change them.**
        {error_report_str}

        ---
        ### PART 2: STANDARD ACTIVITIES (KEEP THESE EXACTLY AS IS)
        {json.dumps(standard_activities, indent=2)}

        ---
        ### PART 3: REFERENCE CONTEXT (Original Flow)
        Use this only to understand the sequence. **DO NOT COPY THE STRUCTURE BLINDLY.**
        {json.dumps(current_json, indent=2)}

        ---
        ### EXAMPLES OF DESIRED GRANULARITY
        {standard_activity_examples}

        ---
        ### OUTPUT FORMAT
        1. **Transformation Plan:** Briefly list the changes (e.g., "Old Name" -> "New Name").
        2. **Final JSON:** A single valid JSON object containing `roles`, `activities`, and `gateways`.


        **Final JSON Structure:**
        ```json
        {{
          "roles": [...],
          "activities": [
            {{ "id": "...", "description": "..." }},
            ...
          ],
          "gateways": [...]
        }}
        ```
        """

    messages = [
        {"role": "system", "content": refinement_prompt},
        {"role": "user",
         "content": "Please fix the granularity issues based on the Error Report. Output the Transformation Plan first, then the JSON."}
    ]

    # 建议稍微调高一点 temperature，让模型有“创造性”去合并/拆分，而不是死板地复制
    response = client.chat.completions.create(
        model='Qwen/Qwen3-235B-A22B-Instruct-2507',
        # Qwen/Qwen3-235B-A22B-Instruct-2507  Qwen/Qwen3-Coder-480B-A35B-Instruct
        messages=messages,
        temperature=0,  # 稍微增加一点随机性，避免死循环复制
        stream=False
    )
    llm_output = response.choices[0].message.content
    print(f"LLM refinement output:\n{llm_output}")

    # --- 增强的 JSON 提取逻辑 ---
    try:
        # 优先寻找 Markdown 代码块
        code_block_pattern = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(code_block_pattern, llm_output, re.DOTALL)

        if matches:
            # 取最后一个代码块（通常是最终结果）
            json_string = matches[-1]
        else:
            # 如果没有代码块，尝试寻找最外层的大括号
            match = re.search(r'\{.*}', llm_output, re.DOTALL)
            if match:
                json_string = match.group(0)
            else:
                raise ValueError("No JSON object found in the LLM output.")

        refined_json = json.loads(json_string)

        # 简单的校验：如果活动数量完全没变，且描述完全没变，可能需要警告
        # (这里可以加额外的逻辑，但先让它跑起来)

        return {
            "processed_json": refined_json,
            "iteration_count": state.iteration_count + 1
        }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"!!! [ERROR] Failed to parse JSON from LLM response. Error: {e} !!!")
        # 打印出错的片段方便调试
        print(f"Debug - Failed JSON string snippet: {llm_output[-500:]}")
        return {
            "processed_json": state.processed_json,
            "iteration_count": state.iteration_count + 1
        }
def should_continue(state: GraphState):
    print("\n--- [ALIGNER_EDGE] Deciding to continue or finish ---")
    if not state.activities_to_refine:
        print("  - Decision: All activities are standard. FINISH.")
        return "end"
    elif state.iteration_count >= 8:
        print("  - Decision: Max iterations reached. FINISH.")
        return "end"
    else:
        print(f"  - Decision: Found {len(state.activities_to_refine)} issues. CONTINUE to refine.")
        return "continue"

workflow = StateGraph(cast(Any, GraphState))
workflow.add_node("critique", cast(Any, critique_node))
workflow.add_node("refine", cast(Any, refine_node))
workflow.set_entry_point("critique")
workflow.add_conditional_edges(
    "critique",
    should_continue,
    {"continue": "refine", "end": END}
)
workflow.add_edge("refine", "critique")
aligner_app = workflow.compile()

@tool(return_direct=True)
def align_activity_granularity(input_data: Union[str, dict]) -> str:
    """
    Takes a BPMN JSON (as a string or a dictionary) and iteratively refines its activities
    until all of them have 'Standard' granularity.
    """
    print("\n--- [ALIGNER_TOOL] Starting Granularity Alignment Loop ---")
    input_json = None
    if isinstance(input_data, dict):
        input_json = input_data
    elif isinstance(input_data, str):
        try:
            start_index = input_data.find('{')
            end_index = input_data.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                clean_json_str = input_data[start_index: end_index + 1]
                input_json = json.loads(clean_json_str)
            else:
                raise json.JSONDecodeError("No valid JSON object found in input string.", input_data, 0)
        except json.JSONDecodeError as e:
            error_message = f"Error: Invalid JSON string input. Could not parse the string. Details: {e}"
            print(error_message)
            return error_message
    else:
        error_message = f"Error: Received unexpected input type: {type(input_data)}"
        print(error_message)
        return error_message

    initial_state = {
        "input_json": input_json,
        "processed_json": input_json,
        "activities_to_refine": [],
        "iteration_count": 0,
    }
    final_state = aligner_app.invoke(cast(Any, initial_state))
    final_aligned_json = final_state["processed_json"]
    return json.dumps(final_aligned_json, indent=2, ensure_ascii=False)

aligner_tools = [align_activity_granularity]
instruction_aligner = """
You are a simple, rule-based dispatcher. Your ONLY function is to take the user's input and immediately call the align_activity_granularity tool with that exact input.

**SHARED MEMORY LOG (Previous Steps):**
---
{memory_log}
---

CRITICAL RULE: You MUST follow the Thought/Action/Action Input format.
Your thought process should be extremely simple. After thinking, you MUST immediately output an 'Action' and 'Action Input'.

EXAMPLE of your ONLY valid response format:

Thought: The user has provided a JSON string. My only job is to call the align_activity_granularity tool with this string.
Action: align_activity_granularity
Action Input: [the complete JSON string provided by the user]
"""
prompt_hub = hub.pull("hwchase17/react")
aligner_template = instruction_aligner + "\n\n" + prompt_hub.template
aligner_prompt = PromptTemplate.from_template(aligner_template)
aligner_agent = create_react_agent(llm=llm, tools=aligner_tools, prompt=aligner_prompt)
aligner_agent_executor = AgentExecutor(agent=aligner_agent, tools=aligner_tools, verbose=True, handle_parsing_errors=True)