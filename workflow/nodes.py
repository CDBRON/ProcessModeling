import json
from typing import Dict, Any

from core import memory
from core.llm import llm
from core.utils import invoke_with_cost_logging
from agents.researcher import researcher_agent_executor
from agents.extractor import perform_extraction # 这是一个函数
from agents.enricher import gateway_inference_agent_executor
from agents.aligner import aligner_agent_executor
from agents.pruner import pruning_agent_executor
from agents.assembler import assembly_agent_executor
from agents.auditor import final_auditor_agent_executor
from workflow.state import TeamProjectState

memory=memory.SharedMemory
def research_node(state: TeamProjectState) -> Dict[str, Any]:
    print("\n\n" + "=" * 20 + " [NODE: RESEARCHER] " + "=" * 20)
    user_request = state["user_request"]
    research_input = {"input": user_request, "memory_log": memory.get_formatted_log()}

    # response = researcher_agent_executor.invoke(research_input)
    # memory.add_entry("ResearcherAgent", research_input, response)

    # 使用我们的包装函数，它会自动处理调用和日志记录
    response = invoke_with_cost_logging(
        researcher_agent_executor,
        research_input,
        "ResearcherAgent",
        memory,
        llm_object=llm  # <--- 新增的参数
    )

    return {"research_report": response['output'], "last_agent_called": "Researcher"}


def extraction_node(state: TeamProjectState) -> Dict[str, Any]:
    print("\n\n" + "=" * 20 + " [NODE: EXTRACTOR] " + "=" * 20)
    report = state["research_report"]
    user_request = state["user_request"]

    extraction_input = {"process_description": report, "user_request": user_request}
    json_str = perform_extraction(report, user_request)
    print("提取的内容为：\n")
    print(json_str)
    memory.add_entry("ExtractionFunction", extraction_input, json_str)

    try:
        skeleton_json = json.loads(json_str)
        return {"skeleton_json": skeleton_json, "last_agent_called": "Extractor"}
    except json.JSONDecodeError:
        return {"skeleton_json": {"error": "Failed to parse JSON"}, "last_agent_called": "Extractor"}


def enrichment_node(state: TeamProjectState) -> Dict[str, Any]:
    print("\n\n" + "=" * 20 + " [NODE: ENRICHER] " + "=" * 20)
    user_request = state.get("user_request", "N/A")
    research_report = state.get("research_report", "N/A")
    skeleton_json_str = json.dumps(state.get('skeleton_json', {}), indent=2)
    memory_log = memory.get_formatted_log()  # 假设 memory 对象在此处可用
    gateway_input = {
        "input": "Based on all the provided context, enrich the Skeleton JSON with gateways.",
        "user_request": user_request,
        "research_report": research_report,
        "skeleton_json": skeleton_json_str,
        "memory_log": memory_log
    }
    # AgentExecutor 现在会正确地填充所有占位符
    # response = gateway_inference_agent_executor.invoke(gateway_input)
    #
    # # 记录到 memory 时，我们可以记录完整的输入
    # memory.add_entry("GatewayInferenceAgent", gateway_input, response)

    # 使用我们的包装函数，它会自动处理调用和日志记录
    response = invoke_with_cost_logging(
        gateway_inference_agent_executor,
        gateway_input,
        "GatewayInferenceAgent",
        memory,
        llm_object=llm  # <--- 新增的参数
    )

    try:
        enriched_json = json.loads(response['output'])
        return {"enriched_json": enriched_json, "last_agent_called": "Enricher"}
    except json.JSONDecodeError:
        return {"enriched_json": {"error": "Failed to parse JSON"}, "last_agent_called": "Enricher"}


def alignment_node(state: TeamProjectState) -> Dict[str, Any]:
    print("\n\n" + "=" * 20 + " [NODE: ALIGNER] " + "=" * 20)
    aligner_input_str = json.dumps(state['enriched_json'])
    aligner_input = {"input": aligner_input_str, "memory_log": memory.get_formatted_log()}
    # response = aligner_agent_executor.invoke(aligner_input)
    # memory.add_entry("GranularityAlignerAgent", aligner_input, response)
    # 使用我们的包装函数，它会自动处理调用和日志记录
    response = invoke_with_cost_logging(
        aligner_agent_executor,
        aligner_input,
        "GranularityAlignerAgent",
        memory,
        llm_object=llm # <--- 新增的参数
    )


    try:
        aligned_json = json.loads(response['output'])
        return {"aligned_json": aligned_json, "last_agent_called": "Aligner"}
    except json.JSONDecodeError:
        return {"aligned_json": {"error": "Failed to parse JSON"}, "last_agent_called": "Aligner"}


def pruning_node(state: TeamProjectState) -> Dict[str, Any]:
    print("\n\n" + "=" * 20 + " [NODE: PRUNER] " + "=" * 20)
    pruning_agent_input_str = f"""
        **Original User Request:**
        ---
        {state['user_request']}
        ---
        **Detailed JSON to Refine:**
        <JSON_START>
        {json.dumps(state['aligned_json'], indent=2)}
        <JSON_END>
        """
    pruning_input = {"input": pruning_agent_input_str, "memory_log": memory.get_formatted_log()}

    response = invoke_with_cost_logging(
        pruning_agent_executor,
        pruning_input,
        "PruningAbstractionAgent",
        memory,
        llm_object=llm # <--- 新增的参数
    )

    try:
        abstracted_json = json.loads(response['output'])
        return {"abstracted_json": abstracted_json, "last_agent_called": "Pruner"}
    except json.JSONDecodeError:
        return {"abstracted_json": {"error": "Failed to parse JSON"}, "last_agent_called": "Pruner"}


def assembly_node(state: TeamProjectState) -> Dict[str, Any]:
    print("\n\n" + "=" * 20 + " [NODE: ASSEMBLER] " + "=" * 20)
    assembly_input_str = json.dumps(state['abstracted_json'])
    assembly_input = {"input": assembly_input_str, "memory_log": memory.get_formatted_log()}


    response = invoke_with_cost_logging(
        assembly_agent_executor,
        assembly_input,
        "ProcessAssemblyAgent",
        memory,
        llm_object=llm  # <--- 新增的参数
    )

    try:
        assembled_ir = json.loads(response['output'])
        return {"assembled_ir": assembled_ir, "last_agent_called": "Assembler"}
    except json.JSONDecodeError:
        return {"assembled_ir": {"error": "Failed to parse JSON"}, "last_agent_called": "Assembler"}


def audit_node(state: TeamProjectState) -> Dict[str, Any]:
    print("\n\n" + "=" * 20 + " [NODE: AUDITOR] " + "=" * 20)
    auditor_input_str = f"""
        **Original User Request:**
        ---
        {state['user_request']}
        ---
        **Process IR to be Audited:**
        <JSON_START>
        {json.dumps(state['assembled_ir'])}
        <JSON_END>
    """
    auditor_input = {"input": auditor_input_str, "memory_log": memory.get_formatted_log()}


    response = invoke_with_cost_logging(
        final_auditor_agent_executor,
        auditor_input,
        "FinalAuditorAgent",
        memory,
        llm_object=llm  # <--- 新增的参数
    )

    try:
        final_ir = json.loads(response['output'])
        return {"final_ir": final_ir, "last_agent_called": "Auditor"}
    except json.JSONDecodeError:
        return {"final_ir": {"error": "Failed to parse JSON"}, "last_agent_called": "Auditor"}