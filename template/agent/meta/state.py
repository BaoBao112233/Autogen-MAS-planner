from typing import TypedDict,Annotated
from template.message.message import BaseMessage
from operator import add

class AgentState(TypedDict):
    input: str
    agent_data:dict
    messages: Annotated[list[BaseMessage],add]
    output: str