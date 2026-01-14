import json
import re
from typing import Union
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_classic import hub
from core.llm import client,llm # client 用于类内部调用

class ProcessAssembler:
    class ProcessAssembler:
        def __init__(self, components_json: dict, llm_client):
            self.llm = llm_client
            self.activities = {act['id']: act for act in components_json.get('activities', [])}
            self.gateways = {gate['id']: gate for gate in components_json.get('gateways', [])}
            self.component_pool = {
                "activities": list(self.activities.values()),
                "gateways": list(self.gateways.values())
            }
            self.used_ids = set()
            self.final_ir = {"process": []}
            self.fix_counter = 0

        def _get_llm_decision(self, prompt: str) -> str:
            messages = [{"role": "system",
                         "content": "You are a logical reasoning engine for BPMN process assembly. Respond concisely and accurately based on the user's query."},
                        {"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model='llm',  # MiniMax/MiniMax-M1-80k     Qwen/Qwen3-235B-A22B
                messages=messages,
                temperature=0,
                stream=False
            )
            return response.choices[0].message.content.strip()

        def _find_start_activity_id(self) -> str:
            print("--- [Assembler] Finding process start point...")
            activity_descriptions = "\n".join(
                [f"- {act['id']}: {act['description']}" for act in self.activities.values()])
            prompt = f"""
            From the following list of business process activities, identify the one that is the most logical starting point for the entire process.
            The starting activity is usually initiated by an external role like a customer or employee.

            Activities:
            {activity_descriptions}

            Respond with ONLY the ID of the starting activity (e.g., "act_1").
            """
            start_id = self._get_llm_decision(prompt)
            match = re.search(r'act[\w_-]*', start_id)
            return match.group(0) if match else list(self.activities.keys())[0]

        def _find_next_happy_path_id(self, current_activity_id: str, remaining_ids: list) -> Union[str, None]:
            if not remaining_ids:
                return None
            current_desc = self.activities[current_activity_id]['description']
            remaining_descs = "\n".join(
                [f"- {act_id}: {self.activities[act_id]['description']}" for act_id in remaining_ids])
            prompt = f"""
            The last step in a business process was: "{current_desc}".
            From the list of remaining available steps below, which one is the most direct and logical next step to continue the 'happy path' (assuming no errors or decisions occur)?
            Remaining Steps:
            {remaining_descs}
            Respond with ONLY the ID of the next activity. If no activity is a logical next step, respond with "None".
            """
            next_id = self._get_llm_decision(prompt)
            if "None" in next_id or not any(rem_id in next_id for rem_id in remaining_ids):
                return None
            match = re.search(r'act[\w_-]*', next_id)
            return match.group(0) if match else None

        def build_process_trunk(self):
            print("--- [Assembler] Building process trunk (Happy Path)...")
            start_id = self._find_start_activity_id()
            if not start_id:
                print("Error: Could not determine a start activity.")
                return
            trunk_ids = []
            current_id = start_id
            remaining_ids = set(self.activities.keys())
            while current_id and current_id in remaining_ids:
                trunk_ids.append(current_id)
                self.used_ids.add(current_id)
                remaining_ids.remove(current_id)
                next_id = self._find_next_happy_path_id(current_id, list(remaining_ids))
                current_id = next_id
            self.final_ir['process'] = [{"type": "activity", "id": act_id} for act_id in trunk_ids]
            print(f"--- [Assembler] Trunk built successfully with {len(trunk_ids)} activities: {trunk_ids}")

        def graft_gateways(self):
            print("--- [Assembler] Grafting gateways onto the trunk...")
            gateways_to_graft = list(self.gateways.values())
            if not gateways_to_graft:
                print("--- [Assembler] No gateways to graft.")
                return
            current_ir_str = json.dumps(self.final_ir, indent=2)
            all_activities_str = json.dumps(list(self.activities.values()), indent=2)
            all_gateways_str = json.dumps(gateways_to_graft, indent=2)
            prompt = f"""
            You are an expert BPMN Process Modeler. Your task is to integrate gateways into a linear process trunk to create a logically structured, nested process model.
            Here is the linear "happy path" trunk of the process:
            <trunk>{current_ir_str}</trunk>
            Here is the complete list of all available activities:
            <activities>{all_activities_str}</activities>
            Here is the complete list of all available gateways that need to be integrated:
            <gateways>{all_gateways_str}</gateways>
            **Your Task:**
            Reconstruct the process flow by correctly placing each gateway from the list into the trunk. For each gateway, create a `branches` array. Each branch should lead to one or more activities from the activity list. The final output must be a single JSON object representing the complete, nested process.
            **Example Output Format:**
            {{
              "process": [
                {{ "type": "activity", "id": "act_1", "description": "......"}},
                {{
                  "type": "exclusiveGateway", "id": "gate_1", "description": "Is the request approved?",
                  "branches": [
                    {{ "condition": "Approved", "sequence": [ {{ "type": "activity", "id": "act_2","description": "......" }} ] }},
                    {{ "condition": "Rejected", "sequence": [ {{ "type": "activity", "id": "act_infer_1","description": "......" }} ] }}
                  ]
                }},
                {{ "type": "activity", "id": "act_3", "description": "......" }}
              ]
            }}
            Now, generate the final, nested JSON for the provided trunk and components.
            Let's think step by step！
            """
            response = client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": "system", "content": prompt},
                    {'role': 'user', 'content': "Tackle the problem in System prompt"}
                ],
                stream=False,
                response_format={"type": "json_object"}
            )
            try:
                self.final_ir = json.loads(response.choices[0].message.content)
                print("--- [Assembler] Gateways grafted successfully using LLM-based reconstruction.")
            except json.JSONDecodeError:
                print("--- [Assembler] Error: LLM failed to generate a valid JSON for gateway grafting.")

        def _collect_used_activity_ids(self, elements: list) -> set:
            used_ids = set()
            if not isinstance(elements, list):
                return used_ids
            for element in elements:
                if element.get('type') == 'activity':
                    used_ids.add(element.get('id'))
                elif 'branches' in element:
                    for branch in element.get('branches', []):
                        used_ids.update(self._collect_used_activity_ids(branch.get('sequence', [])))
            return used_ids

        def _validate_gateways_recursive(self, elements: list):
            if not isinstance(elements, list):
                return
            for element in elements:
                if 'branches' in element:
                    gateway_id = element.get('id')
                    print(f"--- [Validator] Auditing gateway: {gateway_id}...")
                    for i, branch in enumerate(element.get('branches', [])):
                        if not branch.get('sequence'):
                            self.fix_counter += 1
                            fix_act_id = f"act_fix_{self.fix_counter}"
                            fix_act_desc = f"System: Conclude process path from gateway '{gateway_id}'"
                            print(
                                f"  - WARNING: Dangling branch found in gateway '{gateway_id}' (condition: '{branch.get('condition', 'N/A')}').")
                            print(f"  - AUTO-FIX: Creating and connecting new terminal activity '{fix_act_id}'.")
                            self.activities[fix_act_id] = {"id": fix_act_id, "description": fix_act_desc}
                            branch['sequence'] = [{"type": "activity", "id": fix_act_id}]
                        else:
                            self._validate_gateways_recursive(branch.get('sequence', []))

        def _finalize_and_validate(self):
            print("\n--- [Assembler] Finalizing and validating the assembled process...")
            all_used_ids = self._collect_used_activity_ids(self.final_ir.get('process', []))
            all_original_ids = set(self.activities.keys())
            unused_ids = all_original_ids - all_used_ids
            if unused_ids:
                print(
                    f"--- [Validator] The following {len(unused_ids)} activities were not used in the final assembly and are considered pruned:")
                for unused_id in sorted(list(unused_ids)):
                    description = self.activities.get(unused_id, {}).get('description', 'N/A')
                    print(f"  - {unused_id}: {description}")
            else:
                print("--- [Validator] All activities were successfully integrated into the process.")
            self._validate_gateways_recursive(self.final_ir.get('process', []))
            print("--- [Assembler] Validation and auto-fixing complete.")

        def assemble(self) -> dict:
            self.build_process_trunk()
            self.graft_gateways()
            self._finalize_and_validate()
            return self.final_ir


@tool(return_direct=True)
def assemble_process_components(final_abstracted_json_str: str) -> str:
    """
    Use this tool as the final step of the entire pipeline.
    It takes the pruned and abstracted JSON of components and assembles them into a nested,
    structured Intermediate Representation (IR) of the process flow.
    The output of this tool is the definitive, final answer.
    """
    print("\n--- [AssemblyAgent] Starting final process assembly... ---\n")
    try:
        match = re.search(r'{.*}', final_abstracted_json_str, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No valid JSON object found in the input string.", final_abstracted_json_str, 0)
        clean_json_str = match.group(0)
        components_json = json.loads(clean_json_str)
        assembler = ProcessAssembler(components_json, llm)
        final_ir = assembler.assemble()
        return json.dumps(final_ir, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        error_msg = f"Error: Invalid JSON received for assembly. Details: {e}. Raw input was: '{final_abstracted_json_str}'"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during assembly: {e}"
        print(error_msg)
        return error_msg

assembly_tools = [assemble_process_components]
instruction_assembly = """
You are a Process Assembly Orchestrator. Your only job is to take the final, cleaned JSON of process components and pass it to the assemble_process_components tool.
You must not modify the input. Your final thought must be to call the tool with the received JSON.

**SHARED MEMORY LOG (Previous Steps):**
---
{memory_log}
---

Let's think step by step！
"""
prompt_hub = hub.pull("hwchase17/react")
assembly_template = instruction_assembly + "\n\n" + prompt_hub.template
assembly_prompt = PromptTemplate.from_template(assembly_template)
assembly_agent = create_react_agent(
    llm=llm,
    tools=assembly_tools,
    prompt=assembly_prompt
)
assembly_agent_executor = AgentExecutor(
    agent=assembly_agent,
    tools=assembly_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)