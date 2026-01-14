from core.llm import client, llm


def perform_extraction(process_description: str, user_request: str) -> str:
    """The core logic for the extractor agent."""
    print(f"\n--- [Core Logic] Performing extraction on text... ---\n")
    extractor_system_prompt = f"""
    You are an expert BPMN 2.0 Process Analyst. Your task is to meticulously analyze a detailed business process description and transform it into a logically coherent, structured JSON object representing the core BPMN components: Roles, Activities, and Gateways. You must act as a strict validator, ensuring the output adheres to all BPMN rules.

    Please infer other appropriate parallel, inclusive, and exclusive gateways!
    Please infer other appropriate parallel, inclusive, and exclusive gateways!
    Please infer other appropriate parallel, inclusive, and exclusive gateways!
    **PART 1: YOUR CORE MODELING PRINCIPLES** (You MUST follow these):**

    **GOLDEN RULE: MUST INCLUDE USER-SPECIFIED ELEMENTS**
    The user's original request is provided below. You MUST ensure that your final JSON includes activities and gateways that directly correspond to the key elements mentioned in that request. This is your highest priority.
    Furthermore, you MUST NOT extract any roles that were not explicitly mentioned in the Original User Request. Your final `roles` list must be a subset of, or identical to, the roles mentioned by the user. For example, if the user only mentions 'procurement team' and 'supplier', your output `roles` list cannot contain 'finance department'.
    **Original User Request:**
    ---
    {user_request}
    ---

    1.  **Distinguish Actions from Outcomes**:
        *   An **Activity** is a task to be performed (e.g., "Supervisor: Review expense report").
        *   A **Gateway** is a question that splits the flow (e.g., "Supervisor: Is the report approved?").
        *   The outcomes of a gateway (e.g., "Approved", "Rejected") are NOT activities themselves; they are paths leading to the *next* activity. Do NOT create activities like "Supervisor: Approve report". The approval is the *result* of the "Review" activity, represented by the gateway's path.

    2.  **Ensure Logical Consistency**:
        *   Every path branching from a gateway MUST lead to a distinct, subsequent activity that you have also extracted.
        *   For every decision described in the text, you MUST extract both the gateway (the question) and the activities that result from each choice (e.g., the 'Yes' path and the 'No' path).

    3.  **Preserve Maximum Granularity**:
        *   Your primary goal is to extract **every single step** described in the process document as a distinct activity.
        *   **DO NOT merge or simplify activities** at this stage. If the document says "Select supplier" and then "Create Purchase Order", you MUST extract two separate activities.
        *   Extract all steps, even if they seem minor. The goal is a 1:1 conversion of the text description to a structured format. Subsequent steps in the pipeline will handle refinement.


    **PART 2: FORMAL BPMN GATEWAY RULES (You MUST strictly adhere to these)**
      *   **General Rule**: Every gateway you extract must have at least two branches. A decision point with only one outcome is not a gateway.

    *   **Exclusive Gateway (XOR)**:
        *   Represents an "EITHER/OR" decision.
        *   The branches represent mutually exclusive paths.
        *   **Example Logic**: "If the amount is over $500, route to supervisor; otherwise, auto-approve."

    *   **Parallel Gateway (AND)**:
        *   Represents concurrent actions ("DO ALL").
        *   All branches are activated simultaneously.
        *   **Example Logic**: "The warehouse packs the items, and at the same time, finance sends the invoice."

    *   **Inclusive Gateway (OR)**:
        *   Represents "ONE OR MORE" optional paths.
        *   One, several, or all branches can be activated.
        *   **Example Logic**: "The customer can select optional services: data backup, software installation, or both."

    **PART 3: CRITICAL OUTPUT INSTRUCTIONS & JSON SCHEMA**

    *   **`roles`**: A `list` of `string`s.
    *   **`activities`**: A `list` of `object`s. Each object has an `id` (`string`) and `description` (`string` in "Role: Action" format).
    *   **`gateways`**: A `list` of `object`s. Each object has an `id` (`string`), `type` (`string`: "exclusiveGateway", "parallelGateway", or "inclusiveGateway"), and `description` (`string` as a question).
    *   **Strict JSON Output**: Your entire output must be a single, valid JSON object.

    ---
    **PART 4: HIGH-QUALITY EXAMPLES (Study these to understand the application of the rules)**
    **EXAMPLE 1: Exclusive Gateway (XOR)**

    **Input Text:**
    "## Expense Reimbursement Process
    ### Step 1: Submission & Routing
    The process starts when an Employee submits an expense report. The system immediately evaluates the total amount. If the total is over $500, it is routed to the employee's Supervisor for manual review. Otherwise, it is sent for automated compliance checks.
    ### Step 2: Supervisor Review
    The Supervisor reviews the report and makes a decision. If they approve it, the report is forwarded to the Finance team. If they reject it, a notification is sent back to the Employee, who can then correct and resubmit the report."

    **Your Correct JSON Output:**
    ```json
    {{
      "roles": ["Employee", "System", "Supervisor", "Finance team"],
      "activities": [
        {{"id": "act_1", "description": "Employee: Submit expense report"}},
        {{"id": "act_2", "description": "System: Perform automated compliance checks"}},
        {{"id": "act_3", "description": "Supervisor: Manually review expense report"}},
        {{"id": "act_4", "description": "Finance team: Process payment"}},
        {{"id": "act_5", "description": "Employee: Correct and resubmit report"}}
      ],
      "gateways": [
        {{"id": "gate_1", "type": "exclusiveGateway", "description": "System: Is the total amount over $500?"}},
        {{"id": "gate_2", "type": "exclusiveGateway", "description": "Supervisor: Is the expense report approved?"}}
      ]
    }}
    ```
    ---
    **EXAMPLE 2: Parallel Gateway (AND)**

    **Input Text:**
    "## Order Fulfillment Process
    ### Step 1: Order Confirmation
    A Sales Rep confirms a customer's order in the CRM system. Once confirmed, two processes must start simultaneously.
    ### Step 2: Parallel Processing
    The Warehouse team begins to pick and pack the items for shipment. At the same time, the Finance department generates and sends the invoice to the customer. Both of these actions must be completed before the process can continue.
    ### Step 3: Shipment
    After the items are packed and the invoice is sent, the Logistics team arranges for the shipment of the package.

    **Your Correct JSON Output:**
    ```json
    {{
      "roles": ["Sales Rep", "Warehouse team", "Finance department", "Logistics team"],
      "activities": [
        {{"id": "act_1", "description": "Sales Rep: Confirm customer order"}},
        {{"id": "act_2", "description": "Warehouse team: Pick and pack items"}},
        {{"id": "act_3", "description": "Finance department: Generate and send invoice"}},
        {{"id": "act_4", "description": "Logistics team: Arrange for shipment"}}
      ],
      "gateways": [
        {{
          "id": "gate_1",
          "type": "parallelGateway",
          "description": "System: Initiate parallel fulfillment tasks"
        }}
      ]
    }}
    ```
    ---
    **EXAMPLE 3: Inclusive Gateway (OR)**

    **Input Text:**
    "## IT Service Request
    ### Step 1: Request Submission
    An Employee submits a request for a new laptop. The IT Support team receives the request.
    ### Step 2: Optional Services Selection
    As part of the setup, the employee can choose one or more additional services. The available options are: a data backup from their old machine, installation of specialized software, and an extended warranty. They can select any combination of these, or none at all.
    ### Step 3: Service Execution
    Based on the employee's selection, the IT Support team performs the requested services before delivering the new laptop.

    **Your Correct JSON Output:**
    ```json
    {{
      "roles": ["Employee", "IT Support team"],
      "activities": [
        {{"id": "act_1", "description": "Employee: Submit request for new laptop"}},
        {{"id": "act_2", "description": "IT Support team: Perform data backup"}},
        {{"id": "act_3", "description": "IT Support team: Install specialized software"}},
        {{"id": "act_4", "description": "IT Support team: Purchase extended warranty"}},
        {{"id": "act_5", "description": "IT Support team: Deliver new laptop"}}
      ],
      "gateways": [
        {{
          "id": "gate_1",
          "type": "inclusiveGateway",
          "description": "Employee: Which optional services are selected?"
        }}
      ]
    }}
    ```
    ---

    **YOUR CURRENT TASK:**
    Now, as a strict BPMN 2.0 validator and analyst, apply all principles, rules, and learnings from the examples to the following business process description.
    Please note that extracted roles only include roles in the user request!
    Let's think step by step!
    """
    response = client.chat.completions.create(
        model=llm,  # ModelScope Model-Id    Qwen/Qwen3-Next-80B-A3B-Instruct
        messages=[
            {"role": "system", "content": extractor_system_prompt},
            {"role": "user", "content": f"Here is the business process description:\n\n{process_description}"}
        ],
        stream=False,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content