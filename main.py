import os
import json
import datetime
from functools import partial
from langgraph.graph import StateGraph, END

# --- 项目模块导入 ---
from config.settings import Config
from core.utils import TeeOutput
from core.memory import SharedMemory
from workflow.state import TeamProjectState
from workflow.graph import sequential_orchestrator_router

# --- 导入新增的转换与可视化模块 ---
from core.converter import BpmnConverter
from core.postprocess import remove_role_prefix_from_bpmn
from core.visualizer import bpmn_to_svg

# --- 导入各个节点函数 ---
from workflow.nodes import (
    research_node,
    extraction_node,
    enrichment_node,
    alignment_node,
    pruning_node,
    assembly_node,
    audit_node
)


def main():
    # 1. 环境与目录初始化
    # ------------------------------------------------------------------
    # 创建结构化的输出目录
    output_dirs = {
        "logs": "output/logs",
        "json": "output/01_json_ir",
        "xml": "output/02_xml_bpmn",
        "svg": "output/03_svg_viz"
    }
    for path in output_dirs.values():
        if not os.path.exists(path):
            os.makedirs(path)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"{output_dirs['logs']}/execution_{timestamp_str}.txt"

    # 2. 初始化共享组件
    # ------------------------------------------------------------------
    memory = SharedMemory()

    # 3. 构建 LangGraph 工作流
    print("--- [System] Building Workflow Graph... ---")
    workflow = StateGraph(TeamProjectState)

    # 添加节点 (注入 memory)
    workflow.add_node("researcher", partial(research_node, memory=memory))
    workflow.add_node("extractor", partial(extraction_node, memory=memory))
    workflow.add_node("enricher", partial(enrichment_node, memory=memory))
    workflow.add_node("aligner", partial(alignment_node, memory=memory))
    workflow.add_node("pruner", partial(pruning_node, memory=memory))
    workflow.add_node("assembler", partial(assembly_node, memory=memory))
    workflow.add_node("auditor", partial(audit_node, memory=memory))

    # 设置入口点与路由
    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges("researcher", sequential_orchestrator_router, {"extractor": "extractor", "end": END})
    workflow.add_conditional_edges("extractor", sequential_orchestrator_router, {"enricher": "enricher", "end": END})
    workflow.add_conditional_edges("enricher", sequential_orchestrator_router, {"aligner": "aligner", "end": END})
    workflow.add_conditional_edges("aligner", sequential_orchestrator_router, {"pruner": "pruner", "end": END})
    workflow.add_conditional_edges("pruner", sequential_orchestrator_router, {"assembler": "assembler", "end": END})
    workflow.add_conditional_edges("assembler", sequential_orchestrator_router, {"auditor": "auditor", "end": END})
    workflow.add_conditional_edges("auditor", sequential_orchestrator_router, {"end": END})

    app = workflow.compile()

    # 4. 定义用户输入
    # ------------------------------------------------------------------
    user_final_goal = (
        "Please design a 'Parental Leave, Hiring, and Development process' that includes "
        "the roles of 'Parent', 'HR', 'Candidate', and 'Manager'. The key activities are "
        "'maternity leave planning', 'recruitment and onboarding', and 'personal development planning', "
        "and there should be decision points regarding 'leave extension', 'interview preference', "
        "and 'promotion approval'."
    )

    # 5. 执行工作流
    # ------------------------------------------------------------------
    with TeeOutput(log_file_path):
        start_time = datetime.datetime.now()
        print(f"--- [System] Execution Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"--- [Main Flow] User Goal: {user_final_goal} ---")

        initial_state = TeamProjectState(
            user_request=user_final_goal,
            research_report=None, skeleton_json=None, enriched_json=None,
            aligned_json=None, abstracted_json=None, assembled_ir=None,
            final_ir=None, last_agent_called=None
        )

        try:
            final_state = app.invoke(initial_state)
            final_result = final_state.get("final_ir", {"error": "Workflow did not produce a final IR."})
        except Exception as e:
            print(f"\n!!! [CRITICAL ERROR] Workflow execution failed: {e} !!!")
            final_result = {"error": str(e)}

        # 6. 结果处理与转换
        # ------------------------------------------------------------------
        end_time = datetime.datetime.now()
        duration = end_time - start_time

        print(f"\n\n{'=' * 30} [PROJECT COMPLETE] {'=' * 30}")

        if "error" not in final_result:
            # A. 保存 IR JSON
            json_path = f"{output_dirs['json']}/process_{timestamp_str}.json"
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print(f"1. IR JSON saved to: {json_path}")

            try:
                # B. 转换为 BPMN XML
                print("--- [Converter] Converting IR to BPMN XML... ---")
                converter = BpmnConverter(final_result)
                raw_xml = converter.convert()

                # C. 清洗角色前缀
                print("--- [PostProcess] Cleaning Role Prefixes... ---")
                clean_xml = remove_role_prefix_from_bpmn(raw_xml)

                # 保存 XML
                xml_path = f"{output_dirs['xml']}/process_{timestamp_str}.xml"
                with open(xml_path, "w", encoding='utf-8') as f:
                    f.write(clean_xml)
                print(f"2. BPMN XML saved to: {xml_path}")

                # D. 生成 SVG 可视化
                print("--- [Visualizer] Generating SVG... ---")
                svg_base_path = f"{output_dirs['svg']}/process_{timestamp_str}"
                bpmn_to_svg(clean_xml, svg_base_path)
                print(f"3. SVG Image saved to: {svg_base_path}.svg")

            except Exception as e:
                print(f"!!! Error during conversion/visualization: {e}")
        else:
            print(f"--- Workflow terminated with errors. No outputs generated. ---")

        # 7. 统计报告
        time_stats = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration": str(duration)
        }
        memory.add_entry("SystemTimer", "Execution Timing", time_stats)

        print(f"\n{'=' * 30} [EXECUTION TIME REPORT] {'=' * 30}")
        print(f"Total Duration: {str(duration).split('.')[0]}")
        print(f"{'=' * 80}")

        memory.print_full_log()
        memory.calculate_and_print_total_cost()


if __name__ == "__main__":
    main()