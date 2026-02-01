from functools import partial
from langgraph.graph import StateGraph, END
from typing import Literal


from .state import TeamProjectState

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

  
    last_agent = state.get("last_agent_called")
    if last_agent:
        
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
        print(f"--- [ORCHESTRATOR] UNKNOWN STATE. Last agent was {last_agent}. Terminating. ---")
        return "end"


workflow = StateGraph(TeamProjectState)

workflow.add_node("researcher", research_node)
workflow.add_node("extractor", extraction_node)
workflow.add_node("enricher", enrichment_node)
workflow.add_node("aligner", alignment_node)
workflow.add_node("pruner", pruning_node)
workflow.add_node("assembler", assembly_node)
workflow.add_node("auditor", audit_node)


workflow.set_entry_point("researcher")


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
    {"end": END} 
)

orchestrator_app = workflow.compile()
