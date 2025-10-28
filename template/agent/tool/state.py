from typing import TypedDict,Annotated,Optional
from template.message.message import BaseMessage

class AgentState(TypedDict):
    input: str
    token: str
    route: str
    tool_data: dict
    error: str
    output: str