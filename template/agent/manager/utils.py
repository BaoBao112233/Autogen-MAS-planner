"""
Utility functions for Manager Agent
"""
import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def extract_manager_response(content: str) -> Dict[str, Any]:
    """
    Extract structured data from Manager Agent's LLM response
    
    Args:
        content: Raw LLM response content
        
    Returns:
        Dict containing extracted data
    """
    result = {
        'reasoning': '',
        'agent_type': 'direct',
        'confidence': 0.5,
        'explanation': '',
        'direct_answer': ''
    }
    
    try:
        # Extract reasoning
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        # Extract agent_type
        agent_type_match = re.search(r'<agent_type>(.*?)</agent_type>', content, re.DOTALL)
        if agent_type_match:
            agent_type = agent_type_match.group(1).strip().lower()
            if agent_type in ['plan', 'meta', 'tool', 'direct']:
                result['agent_type'] = agent_type
        
        # Extract confidence
        confidence_match = re.search(r'<confidence>(.*?)</confidence>', content, re.DOTALL)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1).strip())
                result['confidence'] = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            except ValueError:
                logger.warning("Could not parse confidence score, using default 0.5")
        
        # Extract explanation
        explanation_match = re.search(r'<explanation>(.*?)</explanation>', content, re.DOTALL)
        if explanation_match:
            result['explanation'] = explanation_match.group(1).strip()
        
        # Extract direct_answer if present
        direct_answer_match = re.search(r'<direct_answer>(.*?)</direct_answer>', content, re.DOTALL)
        if direct_answer_match:
            result['direct_answer'] = direct_answer_match.group(1).strip()
            
    except Exception as e:
        logger.error(f"Error extracting manager response: {str(e)}")
        # Return default values if extraction fails
        result['explanation'] = "Error parsing response"
    
    return result


def classify_query_type(query: str) -> str:
    """
    Classify user query into categories for routing assistance
    
    Args:
        query: User input query
        
    Returns:
        Query type classification
    """
    query_lower = query.lower().strip()
    
    # Selection keywords (check first - most specific)
    selection_keywords = [
        'plan 1', 'plan 2', 'plan 3', 'option 1', 'option 2', 'option 3',
        'choose', 'select', 'pick', 'go with'
    ]
    
    # Check for exact plan selection patterns
    if (query_lower in ['1', '2', '3'] or 
        any(keyword in query_lower for keyword in selection_keywords)):
        return 'selection'
    
    # Planning keywords
    planning_keywords = [
        'plan', 'create', 'setup', 'automate', 'automation', 'configure',
        'install', 'design', 'build', 'smart home', 'schedule'
    ]
    
    # Execution keywords  
    execution_keywords = [
        'turn on', 'turn off', 'set', 'control', 'adjust', 'change',
        'start', 'stop', 'open', 'close', 'increase', 'decrease'
    ]
    
    # Information keywords
    info_keywords = [
        'what is', 'how does', 'explain', 'help', 'info', 'information',
        'tell me', 'describe', 'what can you do'
    ]
    
    # Check for planning keywords
    if any(keyword in query_lower for keyword in planning_keywords):
        return 'planning'
        
    # Check for execution keywords
    if any(keyword in query_lower for keyword in execution_keywords):
        return 'execution'
        
    # Check for information keywords
    if any(keyword in query_lower for keyword in info_keywords):
        return 'information'
    
    # Default classification
    return 'general'


def format_final_response(
    agent_result: Dict[str, Any], 
    agent_type: str, 
    query: str
) -> str:
    """
    Format the final response from delegated agent
    
    Args:
        agent_result: Result from the delegated agent
        agent_type: Type of agent that handled the request
        query: Original user query
        
    Returns:
        Formatted final response
    """
    if agent_type == 'direct':
        return agent_result.get('direct_answer', 'I apologize, but I could not process your request.')
    
    # Format based on agent type
    if agent_type == 'plan':
        return format_plan_response(agent_result, query)
    elif agent_type == 'meta':
        return format_meta_response(agent_result, query)
    elif agent_type == 'tool':
        return format_tool_response(agent_result, query)
    else:
        return str(agent_result.get('output', 'Request processed successfully.'))


def format_plan_response(agent_result: Dict[str, Any], query: str) -> str:
    """Format Plan Agent response"""
    output = agent_result.get('output', '')
    
    if 'waiting_for_selection' in output:
        # Plan options available, format them nicely
        plan_options = agent_result.get('plan_options', {})
        if plan_options:
            response = f"🏠 **Smart Home Automation Plans**\n\n"
            response += f"I've created 3 specialized plans for your request: '{query}'\n\n"
            
            # Security Plan
            if plan_options.get('security_plan'):
                response += "🔒 **Plan 1: Security Priority**\n"
                for i, task in enumerate(plan_options['security_plan'], 1):
                    response += f"   {i}. {task}\n"
                response += "\n"
            
            # Convenience Plan
            if plan_options.get('convenience_plan'):
                response += "🏡 **Plan 2: Convenience Priority**\n"
                for i, task in enumerate(plan_options['convenience_plan'], 1):
                    response += f"   {i}. {task}\n"
                response += "\n"
            
            # Energy Plan
            if plan_options.get('energy_plan'):
                response += "⚡ **Plan 3: Energy Efficiency Priority**\n"
                for i, task in enumerate(plan_options['energy_plan'], 1):
                    response += f"   {i}. {task}\n"
                response += "\n"
            
            response += "Please select which plan you'd like to implement by saying 'Plan 1', 'Plan 2', or 'Plan 3'."
            return response
    
    return output


def format_meta_response(agent_result: Dict[str, Any], query: str) -> str:
    """Format Meta Agent response"""
    output = agent_result.get('output', '')
    agent_data = agent_result.get('agent_data', {})
    
    if agent_data:
        response = f"🧠 **Meta Agent Analysis**\n\n"
        if agent_data.get('Agent Name'):
            response += f"**Agent:** {agent_data['Agent Name']}\n"
        if agent_data.get('Agent Description'):
            response += f"**Description:** {agent_data['Agent Description']}\n"
        if agent_data.get('Tasks'):
            response += f"**Tasks:** {agent_data['Tasks']}\n"
        response += f"\n**Analysis Result:**\n{output}"
        return response
    
    return output


def format_tool_response(agent_result: Dict[str, Any], query: str) -> str:
    """Format Tool Agent response"""
    output = agent_result.get('output', '')
    route = agent_result.get('route', '')
    
    if route and route != 'execute_tool':
        # Add route information for non-execution responses
        response = f"🔧 **Tool Agent ({route.replace('_', ' ').title()})**\n\n{output}"
        return response
    
    return output


def extract_plan_selection(query: str) -> Optional[int]:
    """
    Extract plan selection number from user query
    
    Args:
        query: User input
        
    Returns:
        Plan number (1, 2, or 3) or None
    """
    query_lower = query.lower().strip()
    
    # Check for explicit plan numbers
    if 'plan 1' in query_lower or query_lower == '1':
        return 1
    elif 'plan 2' in query_lower or query_lower == '2':
        return 2
    elif 'plan 3' in query_lower or query_lower == '3':
        return 3
    
    # Check for alternative formats
    if 'option 1' in query_lower or 'first' in query_lower:
        return 1
    elif 'option 2' in query_lower or 'second' in query_lower:
        return 2
    elif 'option 3' in query_lower or 'third' in query_lower:
        return 3
    
    # Check for plan types
    if 'security' in query_lower:
        return 1
    elif 'convenience' in query_lower:
        return 2
    elif 'energy' in query_lower:
        return 3
    
    return None


def validate_agent_result(agent_result: Any, agent_type: str) -> bool:
    """
    Validate the result from a delegated agent
    
    Args:
        agent_result: Result from agent
        agent_type: Type of agent
        
    Returns:
        True if result is valid
    """
    if not agent_result:
        return False
    
    if not isinstance(agent_result, dict):
        return False
    
    # Check for required fields based on agent type
    if agent_type == 'plan':
        return 'output' in agent_result or 'plan_options' in agent_result
    elif agent_type == 'meta':
        return 'output' in agent_result
    elif agent_type == 'tool':
        return 'output' in agent_result
    
    return True


def get_agent_capabilities() -> Dict[str, List[str]]:
    """
    Get capabilities of each agent for routing decisions
    
    Returns:
        Dict mapping agent types to their capabilities
    """
    return {
        'plan': [
            'Create smart home automation plans',
            'Generate 3 priority-based plan options',
            'Handle plan selection and execution',
            'Coordinate complex multi-step workflows'
        ],
        'meta': [
            'Complex reasoning and analysis',
            'Meta-cognitive task handling',
            'Strategic planning decisions',
            'Abstract problem solving'
        ],
        'tool': [
            'Device control operations',
            'MCP tool execution',
            'Smart home device management',
            'Real-time system interactions'
        ],
        'direct': [
            'Simple information queries',
            'Basic question answering',
            'General help and guidance'
        ]
    }