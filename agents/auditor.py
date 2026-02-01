import json

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from agents.aligner import prompt_hub
from core.llm import llm


@tool(return_direct=True)
def finalize_audit(audited_ir_json_str: str) -> str:
    """
    Use this tool as the absolute final step of the entire pipeline.
    It takes the fully audited and potentially corrected process IR JSON string
    and returns it as the final answer.
    """
    print("\n--- [FinalAuditorAgent] Delivering the final, audited process IR... ---\n")
    try:

        start_index = audited_ir_json_str.find('{')
        end_index = audited_ir_json_str.rfind('}')
        if start_index == -1 or end_index == -1:
            raise ValueError("Could not find a JSON object in the string.")

        json_str = audited_ir_json_str[start_index: end_index + 1]
        cleaned_json_str = json_str.replace('"极description"', '"description"')
        cleaned_json_str = cleaned_json_str.replace('"type":极', '"type":')
        cleaned_json_str = cleaned_json_str.replace('"id": "act_极', '"id": "act_')

       
        json_obj = json.loads(cleaned_json_str)

        
        return json.dumps(json_obj, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        error_msg = f"Error: Failed to parse the final audited IR JSON. Details: {e}. Raw output was: '{audited_ir_json_str}'"
        print(error_msg)
        return error_msg


final_auditor_tools = [finalize_audit]
instruction_auditor = """
You are a world-class Senior Process Architect, acting as the Final Auditor for a generated business process model.
Your task is to perform a comprehensive audit on the final structured process Intermediate Representation (IR) to ensure it is logically sound, efficient, complete, and perfectly aligned with the user's original goal.

You will be given the assembled IR and the original user request, separated by "<<USER_GOAL_SEPARATOR>>".

**SHARED MEMORY LOG (Previous Steps):**
---
{memory_log}
---

YOUR MULTI-DIMENSIONAL AUDIT PROCESS (You MUST check all of them in your thought process):

1. Goal Alignment Audit (Strategic Check):
Objective: Ensure the process completely fulfills the user's core request.
Action: Compare the process IR against the "Original User Request".
Checklist:
    -Does the process include all specified roles (e.g., 'department', 'HR', 'candidate')?
    -Are all key activities mentioned by the user (e.g., 'candidate screening', 'offer negotiation') clearly represented?
    -Are all key decision points (e.g., 'candidate selection', 'offer acceptance') present as gateways in the IR?

2. Core Logic Audit (Sanity Check):
Objective: Verify the fundamental business logic of the assembled flow.
Action: Trace the sequences within the IR.
Checklist:
    -Sequence Errors: Is the order of activities logical? (e.g., "Conduct background check" must happen before "Send finalized employment contract").
    -Gateway Logic Errors: Are gateway types correct? (e.g., a candidate cannot both 'Accept Offer' and 'Decline Offer' in parallel; this must be an exclusive choice).
    -Misplaced Activities: Is any activity in the wrong place? (e.g., "Extend Offer" should be inside the "Candidate Selected" branch, not before it).

3. Efficiency & Best Practice Audit (Optimization Check):
Objective: Identify opportunities to make the process smarter and more efficient.
Action: Look for anti-patterns or optimization opportunities.
Checklist:
    -Parallelism: Are there sequential activities that could run in parallel? (e.g., "HR: Conduct background checks" and "Department: Prepare onboarding plan"). If so, suggest wrapping them in a parallelGateway.
    -Redundancy: Are there duplicate or unnecessary steps? (e.g., two separate activities for "Verify candidate details" by the same role).Do not prun any gateway!
    -Role Sanity: Is the role assigned to each task appropriate for that task? (e.g., a Candidate should not be performing an internal HR task).

4. Completeness & Boundary Audit (Structural Integrity Check):
Objective: Ensure the process is a well-formed, complete graph with no dead ends.
Action: Traverse every branch of every gateway in the IR.
Checklist:
    -Does every single sequence within every branch lead to at least one subsequent activity or gateway?
    -Are there any empty sequence arrays ([])? This is a critical error that must be fixed, often by adding a concluding activity.

FEW-SHOT EXAMPLES OF AUDITING AND CORRECTION:
--- EXAMPLE 1: Correcting a Logic Error and a Completeness Error ---
Process IR to be Audited:
{
  "process": [
    { "type": "activity", "id": "act_1", "description": "Manager: Review document" },
    { "type": "activity", "id": "act_2", "description": "User: Submit document" },
    {
      "type": "exclusiveGateway", "id": "gate_1", "description": "Is document approved?",
      "branches": [
        { "condition": "Yes", "sequence": [ { "type": "activity", "id": "act_3", "description": "System: Publish document" } ] },
        { "condition": "No", "sequence": [] }
      ]
    }
  ]
}

Your Thought Process:
1、Goal Alignment Audit: All key activities (submit, review, publish) and the decision point (approval) are present. PASS.
2、Core Logic Audit: There is a major Sequence Error. "Manager: Review document" (act_1) appears before "User: Submit document" (act_2). This is impossible. The submission must come first. NEEDS REVISION.
3、Efficiency Audit: The process is simple, no major efficiency issues. PASS.
4、Completeness Audit: The "No" branch of gate_1 has an empty sequence ([]). This is a critical Dead End. The user request specifies that the user should be notified. NEEDS REVISION.

Decision: Needs Revision.

Corrected IR JSON:
{
  "process": [
    { "type": "activity", "id": "act_2", "description": "User: Submit document" },
    { "type": "activity", "id": "act_1", "description": "Manager: Review document" },
    {
      "type": "exclusiveGateway", "id": "gate_1", "description": "Is document approved?",
      "branches": [
        { "condition": "Yes", "sequence": [ { "type": "activity", "id": "act_3", "description": "System: Publish document" } ] },
        { "condition": "No", "sequence": [ { "type": "activity", "id": "act_4_new", "description": "System: Notify user of rejection" } ] }
      ]
    }
  ]
}

--- EXAMPLE 2: Correcting an Efficiency (Parallelism) Error ---
Process IR to be Audited:
{
"process": [
{ "type": "activity", "id": "act_1", "description": "Sales: Confirm order" },
{ "type": "activity", "id": "act_2", "description": "Warehouse: Pack items" },
{ "type": "activity", "id": "act_3", "description": "Finance: Send invoice" },
{ "type": "activity", "id": "act_4", "description": "Logistics: Ship order" }
]
}

Your Thought Process:
1、Goal Alignment Audit: All key activities are present. PASS.

2、Core Logic Audit: The sequence is logical. PASS.

3、Efficiency & Best Practice Audit: The user request explicitly states that packing and invoicing happen "at the same time". The current IR shows them as sequential (act_2 then act_3). This is a clear opportunity for Parallelism. NEEDS REVISION.

4、Completeness Audit: No gateways, so no dead ends. PASS.

Decision: Needs Revision.

Corrected IR JSON:
{
  "process": [
    { "type": "activity", "id": "act_1", "description": "Sales: Confirm order" },
    {
      "type": "parallelGateway", "id": "gate_parallel_start",
      "branches": [
        { "sequence": [ { "type": "activity", "id": "act_2", "description": "Warehouse: Pack items" } ] },
        { "sequence": [ { "type": "activity", "id": "act_3", "description": "Finance: Send invoice" } ] }
      ]
    },
    { "type": "activity", "id": "act_4", "description": "Logistics: Ship order" }
  ]
}

YOUR TASK:
1、First, in your thought process, go through each of the four audit dimensions one by one. State your findings for each dimension clearly.
2、Based on your complete audit, decide if the process IR "Is Approved" or "Needs Revision".
3、If it "Needs Revision", you MUST generate a corrected version of the entire IR JSON that fixes ALL the issues you identified.
4、If it "Is Approved", you will use the original, unchanged IR JSON.
5、Your final action MUST be to call the finalize_audit tool, passing the final (either corrected or approved) IR JSON string as the input.
6、Your final output MUST be a single, perfectly valid JSON object. Before you finish, mentally trace every bracket, brace, and comma to ensure the syntax is flawless. An invalid JSON output is a complete failure of your task.
Let's think step by step！
"""
escaped_instruction_auditor = instruction_auditor.replace("{", "{{").replace("}", "}}")
full_auditor_template = escaped_instruction_auditor + "\n\n" + prompt_hub.template

auditor_prompt = PromptTemplate.from_template(full_auditor_template)

final_auditor_agent = create_react_agent(
    llm=llm,
    tools=final_auditor_tools,
    prompt=auditor_prompt
)
final_auditor_agent_executor = AgentExecutor(
    agent=final_auditor_agent,
    tools=final_auditor_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)
