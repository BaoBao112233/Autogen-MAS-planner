"""
State definitions for Manager Agent
"""
from typing import TypedDict, List, Dict, Any, Optional
from template.message.message import HumanMessage, SystemMessage


class ManagerState(TypedDict):
    """State for Manager Agent workflow"""
    input: str                          # User query/input
    token: str                          # Authentication token
    messages: List[Any]                 # Message history
    route: str                          # Current route (plan/meta/direct)
    reasoning_result: Dict[str, Any]    # Result from reasoning step
    agent_type: str                     # Type of agent to delegate to
    delegation_result: Dict[str, Any]   # Result from delegated agent
    final_answer: str                   # Final formatted answer
    output: str                         # Final output
    context: Dict[str, Any]             # Additional context
    needs_planning: bool                # Whether user needs planning
    needs_execution: bool               # Whether user needs execution
    query_type: str                     # Type of query (planning/execution/info)
    confidence_score: float             # Confidence in routing decision