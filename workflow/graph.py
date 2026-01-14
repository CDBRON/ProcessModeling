from functools import partial
from langgraph.graph import StateGraph, END
from typing import Literal

# 导入状态定义
from .state import TeamProjectState

# 导入具体的节点函数
from .nodes import (
    research_node,
    extraction_node,
    enrichment_node,
    alignment_node,
    pruning_node,
    assembly_node,
    audit_node
)

def sequential_orchestrator_router(state: TeamProjectState) -> str:
    print("\n\n" + "--- [ORCHESTRATOR] Checking assembly line status... ---")

    # 质量检查点：检查上一步是否有错误
    last_agent = state.get("last_agent_called")
    if last_agent:
        # 动态获取上一步的输出键
        # e.g., 'Researcher' -> 'research_report', 'Extractor' -> 'skeleton_json'
        output_key_map = {
            "Researcher": "research_report", "Extractor": "skeleton_json",
            "Enricher": "enriched_json", "Aligner": "aligned_json",
            "Pruner": "abstracted_json", "Assembler": "assembled_ir",
            "Auditor": "final_ir"
        }
        latest_output_key = output_key_map.get(last_agent)


        latest_output = state.get(latest_output_key)

        if isinstance(latest_output, dict) and "error" in latest_output:
            print(f"--- [ORCHESTRATOR] FATAL ERROR from {last_agent}. Terminating workflow. ---")
            return "end"

    # 严格的顺序调度逻辑
    if last_agent is None:
        print("--- [ORCHESTRATOR] Decision: Start with Research. ---")
        return "researcher"
    elif last_agent == "Researcher":
        print("--- [ORCHESTRATOR] Decision: Proceed to Extraction. ---")
        return "extractor"
    elif last_agent == "Extractor":
        print("--- [ORCHESTRATOR] Decision: Proceed to Enrichment. ---")
        return "enricher"
    elif last_agent == "Enricher":
        print("--- [ORCHESTRATOR] Decision: Proceed to Alignment. ---")
        return "aligner"
    elif last_agent == "Aligner":
        print("--- [ORCHESTRATOR] Decision: Proceed to Pruning. ---")
        return "pruner"
    elif last_agent == "Pruner":
        print("--- [ORCHESTRATOR] Decision: Proceed to Assembly. ---")
        return "assembler"
    elif last_agent == "Assembler":
        print("--- [ORCHESTRATOR] Decision: Proceed to Final Audit. ---")
        return "auditor"
    elif last_agent == "Auditor":
        print("--- [ORCHESTRATOR] Decision: All steps complete. Finishing project. ---")
        return "end"
    else:
        # 容错：如果出现未知状态，则终止
        print(f"--- [ORCHESTRATOR] UNKNOWN STATE. Last agent was {last_agent}. Terminating. ---")
        return "end"


# 4. 构建并编译 Orchestrator 的工作流程图
workflow = StateGraph(TeamProjectState)

# 添加所有“工人”节点
workflow.add_node("researcher", research_node)
workflow.add_node("extractor", extraction_node)
workflow.add_node("enricher", enrichment_node)
workflow.add_node("aligner", alignment_node)
workflow.add_node("pruner", pruning_node)
workflow.add_node("assembler", assembly_node)
workflow.add_node("auditor", audit_node)

# 设置起始点
workflow.set_entry_point("researcher")

# 将所有节点连接到 Orchestrator 的决策中心
# 注意：这里我们不再需要一个单独的 router 节点，
# 我们可以直接将节点线性连接起来，这更符合顺序执行的理念。
# 但为了保持“Orchestrator”的概念，我们使用条件边，其逻辑是固定的。
workflow.add_conditional_edges(
    "researcher",
    sequential_orchestrator_router,
    {"extractor": "extractor", "end": END}
)
workflow.add_conditional_edges(
    "extractor",
    sequential_orchestrator_router,
    {"enricher": "enricher", "end": END}
)
workflow.add_conditional_edges(
    "enricher",
    sequential_orchestrator_router,
    {"aligner": "aligner", "end": END}
)
workflow.add_conditional_edges(
    "aligner",
    sequential_orchestrator_router,
    {"pruner": "pruner", "end": END}
)
workflow.add_conditional_edges(
    "pruner",
    sequential_orchestrator_router,
    {"assembler": "assembler", "end": END}
)
workflow.add_conditional_edges(
    "assembler",
    sequential_orchestrator_router,
    {"auditor": "auditor", "end": END}
)
workflow.add_conditional_edges(
    "auditor",
    sequential_orchestrator_router,
    {"end": END}  # 审计完成后，流程结束
)

# 编译成可执行的应用
orchestrator_app = workflow.compile()