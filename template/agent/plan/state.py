from typing import TypedDict, Annotated, Optional
from template.message.message import BaseMessage
from operator import add

def subtract(l1: list[str], l2: list[str]) -> list[str]:
    return list(set(l1) - set(l2))

class PlanState(TypedDict, total=False):
    input: str
    plan_type: str
    plan_status: str
    plan: list[str]
    plan_options: dict  # Store plan options for user selection
    needs_user_selection: bool  # Flag to indicate if waiting for user selection
    selected_plan_id: Optional[int]  # User's selected plan ID
    token: Optional[str]  # Authentication token for MCP tools
    output: str

class UpdateState(TypedDict):
    plan:str
    current:str
    responses:Annotated[list[str],add]
    pending: list[str]
    completed: list[str]
    output:str
    messages: Annotated[list[BaseMessage],add]