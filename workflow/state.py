from typing import TypedDict, Dict

class TeamProjectState(TypedDict):
    user_request: str
    research_report: str
    skeleton_json: Dict
    enriched_json: Dict
    aligned_json: Dict
    abstracted_json: Dict
    assembled_ir: Dict
    final_ir: Dict
    last_agent_called: str