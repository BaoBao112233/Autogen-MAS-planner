"""
Manager Agent - Central coordinator for the Multi-Agent System (MAS)

The Manager Agent serves as the entry point and orchestrator for the entire MAS system.
It analyzes user queries, makes routing decisions, and coordinates interactions between
specialized agents (Plan Agent, Meta Agent, Tool Agent).
"""

from template.agent import BaseAgent
from template.configs.environments import env
from template.agent.manager.state import ManagerState
from template.agent.manager.prompt import MANAGER_PROMPT
from template.agent.manager.utils import (
    extract_manager_response,
    classify_query_type,
    format_final_response,
    extract_plan_selection,
    validate_agent_result,
    get_agent_capabilities
)
from template.message.message import HumanMessage, SystemMessage
from template.message.converter import convert_messages_list
from template.agent.histories import RedisSupportChatHistory

from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END, START
from termcolor import colored
import logging
import time
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class ManagerAgent(BaseAgent):
    """
    Manager Agent - Central coordinator for the Multi-Agent System
    
    Responsibilities:
    1. Analyze and classify user queries
    2. Make intelligent routing decisions
    3. Delegate tasks to appropriate specialized agents
    4. Coordinate multi-agent workflows
    5. Format and deliver final responses
    """
    
    def __init__(self, 
                 session_id: str,
                 conversation_id: str,
                 model: str = "gemini-2.5-pro", 
                 temperature: float = 0.2, 
                 max_iteration: int = 5,
                 verbose: bool = False,
                 llm=None):
        super().__init__()
        
        self.name = "Manager Agent"
        
        # Chat history for conversation context
        self.session_id = session_id
        self.chat_history = RedisSupportChatHistory(
            session_id=session_id,
            conversation_id=conversation_id,
            ttl=env.TTL_SECONDS
        )
        
        # LLM configuration
        if llm:
            self.llm = llm
        else:
            self.llm = ChatVertexAI(
                model_name=model,
                temperature=temperature,
                project=env.GOOGLE_CLOUD_PROJECT,
                location=env.GOOGLE_CLOUD_LOCATION
            )
        
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.system_prompt = MANAGER_PROMPT
        
        # Lazy-loaded sub-agents
        self._plan_agent = None
        self._meta_agent = None
        self._tool_agent = None
        
        # Session state for plan persistence
        self._cached_plan_options = {}
        
        # Create execution graph
        self.graph = self.create_graph()
        
        if self.verbose:
            logger.info(f"✅ {self.name} initialized successfully")
    
    @property
    def plan_agent(self):
        """Lazy load Plan Agent"""
        if self._plan_agent is None:
            from template.agent.plan import PlanAgent
            self._plan_agent = PlanAgent(verbose=self.verbose)
            
            # Initialize async components
            try:
                import asyncio
                import threading
                
                def run_async_init():
                    # Create new event loop in separate thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._plan_agent.init_async())
                        logger.info("📋 Plan Agent initialized with LLM")
                    except Exception as e:
                        logger.warning(f"⚠️ Plan Agent async init failed: {e}")
                    finally:
                        loop.close()
                
                # Run in separate thread to avoid event loop conflict
                thread = threading.Thread(target=run_async_init)
                thread.start()
                thread.join(timeout=10)  # Wait max 10 seconds
                
                if thread.is_alive():
                    logger.warning("⚠️ Plan Agent async init timeout")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not init Plan Agent async: {e}")
                logger.info("🧪 Plan Agent will use fallback mode")
            
            logger.info("📋 Plan Agent loaded")
        return self._plan_agent
    
    @property
    def meta_agent(self):
        """Lazy load Meta Agent"""
        if self._meta_agent is None:
            from template.agent.meta import MetaAgent
            self._meta_agent = MetaAgent(
                llm=self.llm,
                verbose=self.verbose
            )
            logger.info("🧠 Meta Agent loaded")
        return self._meta_agent
    
    @property
    def tool_agent(self):
        """Lazy load Tool Agent"""
        if self._tool_agent is None:
            from template.agent.tool import ToolAgent
            self._tool_agent = ToolAgent(verbose=self.verbose)
            # Initialize ToolAgent async components
            try:
                import asyncio
                import threading
                
                def run_async_init():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._tool_agent.init_async())
                    finally:
                        loop.close()
                
                thread = threading.Thread(target=run_async_init)
                thread.start()
                thread.join(timeout=10)
                
                if thread.is_alive():
                    logger.warning("⚠️ ToolAgent async init timeout")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not init ToolAgent async: {e}")
            
            logger.info("🔧 Tool Agent loaded")
        return self._tool_agent
    
    def analyze_query(self, state: ManagerState) -> ManagerState:
        """
        Analyze user query and determine routing strategy with conversation context
        """
        user_input = state.get('input', '')
        
        if self.verbose:
            logger.info(f"🔍 Analyzing query: {user_input}")
        
        # Build messages with conversation history
        messages = []
        
        # Add system prompt
        messages.append(SystemMessage(self.system_prompt))
        
        # Add conversation history for context
        if self.chat_history.messages:
            # Add last few messages for context (limit to prevent token overflow)
            recent_messages = self.chat_history.messages[-6:]  # Last 6 messages (3 exchanges)
            
            # Convert chat history messages to our custom format for consistency
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                    if hasattr(msg, '__class__'):
                        if 'Human' in msg.__class__.__name__:
                            messages.append(HumanMessage(f"Previous User: {content}"))
                        elif 'AI' in msg.__class__.__name__:
                            messages.append(SystemMessage(f"Previous AI: {content}"))
            
            if self.verbose:
                logger.info(f"📚 Using {len(recent_messages)} previous messages for context")
        
        # Add current user query
        messages.append(HumanMessage(f"Current User Query: {user_input}"))
        
        try:
            # Convert and invoke LLM
            lc_messages = convert_messages_list(messages)
            llm_response = self.llm.invoke(lc_messages)
            
            # Extract structured response
            reasoning_result = extract_manager_response(llm_response.content)
            
            if self.verbose:
                logger.info(f"🎯 Routing decision: {reasoning_result['agent_type']} (confidence: {reasoning_result['confidence']:.2f})")
                logger.info(f"📝 Reasoning: {reasoning_result['reasoning'][:100]}...")
            
            # Classify query type for additional context
            query_type = classify_query_type(user_input)
            
            # Update state
            return {
                **state,
                'messages': messages,
                'reasoning_result': reasoning_result,
                'agent_type': reasoning_result['agent_type'],
                'query_type': query_type,
                'confidence_score': reasoning_result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in query analysis: {str(e)}")
            
            # Enhanced fallback routing that considers context
            fallback_type = self._context_aware_fallback_routing(user_input)
            
            return {
                **state,
                'messages': messages,
                'reasoning_result': {
                    'reasoning': f'Fallback routing due to analysis error: {str(e)}',
                    'agent_type': fallback_type,
                    'confidence': 0.3,
                    'explanation': f'Using fallback routing to {fallback_type} agent'
                },
                'agent_type': fallback_type,
                'query_type': classify_query_type(user_input),
                'confidence_score': 0.3
            }
    
    def _context_aware_fallback_routing(self, query: str) -> str:
        """Enhanced fallback routing that considers conversation context"""
        query_lower = query.lower().strip()
        
        # Priority 1: Check for explicit plan creation keywords (highest priority)
        if any(word in query_lower for word in ['create plan', 'make plan', 'plan', 'create', 'setup', 'automate']):
            return 'plan'
        
        # Priority 2: Check conversation history for context clues
        if self.chat_history.messages:
            recent_messages = self.chat_history.messages[-4:]  # Last 4 messages for context
            
            # Look for patterns indicating this is a response to an AI question
            for i, msg in enumerate(recent_messages):
                if hasattr(msg, 'content') and 'AI' in msg.__class__.__name__:
                    content = msg.content.lower()
                    
                    # AI asked about device control parameters
                    if any(word in content for word in ['mode', 'temperature', 'brightness', 'color', 'speed']):
                        # User is likely providing parameters -> route to tool
                        if any(word in query_lower for word in ['default', 'auto', 'high', 'low', 'medium', 'on', 'off']):
                            return 'tool'
                    
                    # AI presented plan options
                    if any(word in content for word in ['plan', 'option', 'choose', 'select']):
                        # User is likely selecting a plan
                        if any(word in query_lower for word in ['1', '2', '3', 'plan', 'option', 'first', 'second', 'third']):
                            return 'plan'
                    
                    # AI asked for clarification about device control
                    if any(word in content for word in ['which', 'what', 'where', 'how']):
                        # Check if previous context was about devices
                        if i > 0 and hasattr(recent_messages[i-1], 'content'):
                            prev_content = recent_messages[i-1].content.lower()
                            if any(word in prev_content for word in ['turn', 'set', 'control', 'adjust', 'device']):
                                return 'tool'
        
        # Priority 3: Fall back to standard keyword-based routing
        return self._fallback_routing(query)
    
    def _fallback_routing(self, query: str) -> str:
        """Simple fallback routing when LLM analysis fails"""
        query_lower = query.lower()
        query_type = classify_query_type(query)
        
        # Priority 1: Plan creation (highest priority)
        if any(word in query_lower for word in ['create plan', 'plan', 'create', 'setup', 'automate', 'automation']):
            return 'plan'
        
        # Priority 2: Plan selection
        elif query_type == 'selection':
            return 'plan'
        
        # Priority 3: Device control
        elif any(word in query_lower for word in ['turn', 'set', 'control', 'adjust']):
            return 'tool'
        
        # Priority 4: Analysis requests (be more specific to avoid conflicts)
        elif any(word in query_lower for word in ['analyze', 'analysis', 'evaluate', 'assessment', 'think', 'reason']):
            # Make sure it's not a plan request
            if not any(word in query_lower for word in ['plan', 'create', 'setup', 'automate']):
                return 'meta'
        
        # Default to direct for information questions
        else:
            return 'direct'
    
    def route_to_agent(self, state: ManagerState) -> ManagerState:
        """
        Route request to appropriate specialized agent
        """
        agent_type = state.get('agent_type', 'direct')
        user_input = state.get('input', '')
        
        if self.verbose:
            logger.info(f"🚀 Routing to {agent_type} agent")
        
        try:
            if agent_type == 'direct':
                # Handle direct responses
                reasoning_result = state.get('reasoning_result', {})
                direct_answer = reasoning_result.get('direct_answer')
                
                if not direct_answer:
                    # Provide default helpful response based on query type
                    query_type = state.get('query_type', 'general')
                    if query_type == 'information':
                        direct_answer = """🏠 **Smart Home Information**

A smart home is a residence equipped with internet-connected devices that enable remote monitoring and management of appliances and systems, such as lighting, heating, security, and entertainment systems.

**Key Benefits:**
• **Convenience**: Control devices remotely via apps or voice commands
• **Energy Efficiency**: Automated scheduling and optimization
• **Security**: Real-time monitoring and alerts
• **Comfort**: Personalized automation based on preferences

**How I Can Help:**
• Create customized automation plans (security, convenience, or energy-focused)
• Control smart home devices in real-time
• Provide guidance on smart home setup and optimization

What would you like to know more about?"""
                    else:
                        direct_answer = """🤖 **Smart Home Assistant**

I'm your Multi-Agent Smart Home Assistant! I can help you with:

🏗️ **Planning**: Create comprehensive smart home automation plans
🔧 **Control**: Manage and control your smart devices
🧠 **Analysis**: Provide strategic insights for home automation
📋 **Guidance**: Answer questions and provide recommendations

**Popular Commands:**
• "Create a smart home plan for my bedroom"
• "Turn on the living room lights"
• "Set the air conditioner to 22 degrees"
• "What's the best way to automate my home?"

How can I assist you today?"""
                
                delegation_result = {
                    'output': direct_answer,
                    'direct_answer': direct_answer,  # Add this for format_final_response
                    'agent_type': 'direct',
                    'success': True
                }
                
            elif agent_type == 'plan':
                # Load cached plan options from context if available
                context = state.get('context', {})
                if 'cached_plan_options' in context:
                    self._cached_plan_options = context['cached_plan_options']
                    if self.verbose:
                        logger.info(f"📋 Loaded cached plan options: {list(self._cached_plan_options.keys())}")
                
                # Check if this is a plan selection
                plan_selection = extract_plan_selection(user_input)
                query_type = state.get('query_type', '')
                
                # Extract token from state for PlanAgent authentication
                token = state.get('token', '')
                
                if (plan_selection and self._cached_plan_options) or query_type == 'selection':
                    # User is selecting from previously generated plans
                    if plan_selection:
                        delegation_result = self.plan_agent.invoke(
                            user_input, 
                            selected_plan_id=plan_selection,
                            plan_options=self._cached_plan_options,
                            token=token
                        )
                    else:
                        # Try to extract plan number from query_type selection
                        fallback_selection = extract_plan_selection(user_input) or 1
                        delegation_result = self.plan_agent.invoke(
                            user_input,
                            selected_plan_id=fallback_selection,
                            plan_options=self._cached_plan_options,
                            token=token
                        )
                else:
                    # New planning request
                    delegation_result = self.plan_agent.invoke(user_input, token=token)
                    
                    # Cache plan options for future selections
                    if delegation_result.get('plan_options'):
                        self._cached_plan_options = delegation_result['plan_options']
                
            elif agent_type == 'meta':
                # Route to Meta Agent
                delegation_result = self.meta_agent.invoke(user_input)
                
            elif agent_type == 'tool':
                # Route to Tool Agent with token
                token = state.get('token', '')
                # Ensure input is in proper format with token
                tool_input = {
                    'input': user_input,
                    'token': token
                }
                delegation_result = self.tool_agent.invoke(tool_input)
                
            else:
                # Unknown agent type
                delegation_result = {
                    'output': f'Unknown agent type: {agent_type}',
                    'success': False,
                    'error': f'Agent type {agent_type} not recognized'
                }
            
            # Validate result
            if not validate_agent_result(delegation_result, agent_type):
                logger.warning(f"⚠️ Invalid result from {agent_type} agent")
                delegation_result = {
                    'output': 'I apologize, but there was an issue processing your request.',
                    'success': False,
                    'error': f'Invalid result from {agent_type} agent'
                }
            
            return {
                **state,
                'delegation_result': delegation_result
            }
            
        except Exception as e:
            logger.error(f"❌ Error routing to {agent_type} agent: {str(e)}")
            
            delegation_result = {
                'output': f'I encountered an error while processing your request: {str(e)}',
                'success': False,
                'error': str(e)
            }
            
            return {
                **state,
                'delegation_result': delegation_result
            }
    
    def finalize_response(self, state: ManagerState) -> ManagerState:
        """
        Format and finalize the response for the user
        """
        delegation_result = state.get('delegation_result', {})
        agent_type = state.get('agent_type', 'direct')
        user_input = state.get('input', '')
        
        if self.verbose:
            logger.info(f"📝 Finalizing response from {agent_type} agent")
        
        try:
            # Format the final response
            final_answer = format_final_response(delegation_result, agent_type, user_input)
            
            # Add metadata for debugging if verbose
            if self.verbose and agent_type != 'direct':
                reasoning = state.get('reasoning_result', {}).get('reasoning', '')
                confidence = state.get('confidence_score', 0.0)
                
                debug_info = f"\n\n---\n*Debug Info: Routed to {agent_type} agent (confidence: {confidence:.2f})*"
                final_answer += debug_info
            
            return {
                **state,
                'final_answer': final_answer,
                'output': final_answer
            }
            
        except Exception as e:
            logger.error(f"❌ Error finalizing response: {str(e)}")
            
            fallback_response = "I apologize, but I encountered an error while preparing my response. Please try again."
            
            return {
                **state,
                'final_answer': fallback_response,
                'output': fallback_response
            }
    
    def controller(self, state: ManagerState) -> str:
        """
        Control the flow of execution in the graph
        """
        # Simple linear flow for now
        # Can be extended for more complex routing logic
        return 'route'
    
    def create_graph(self) -> StateGraph:
        """
        Create the execution graph for the Manager Agent
        """
        graph = StateGraph(ManagerState)
        
        # Add nodes
        graph.add_node('analyze', self.analyze_query)
        graph.add_node('delegate', self.route_to_agent)
        graph.add_node('finalize', self.finalize_response)
        
        # Set entry point
        graph.add_edge(START, 'analyze')
        
        # Add edges
        graph.add_edge('analyze', 'delegate')
        graph.add_edge('delegate', 'finalize')
        graph.add_edge('finalize', END)
        
        return graph.compile(debug=False)
    
    def invoke(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for the Manager Agent
        
        Args:
            input_data: User query/input
            context: Additional context including cached plan options
            
        Returns:
            Dict containing the final response and metadata
        """
        start_time = time.time()
        
        if self.verbose:
            print(f'Entering ' + colored(self.name, 'black', 'on_white'))
            logger.info(f"📥 Processing input: {input_data}")
        
        # Extract user message
        if isinstance(input_data, dict):
            user_message = input_data.get('message', '') or input_data.get('input', '')
        else:
            user_message = input_data
        
        # Save user message to chat history
        self.chat_history.add_user_message(user_message)
        
        # Initialize state
        state = {
            'input': user_message,
            'token': input_data.get('token', '') if isinstance(input_data, dict) else '',
            'messages': [],
            'route': '',
            'context': context or {},
            'reasoning_result': {},
            'agent_type': 'direct',
            'delegation_result': {},
            'final_answer': '',
            'output': '',
            'needs_planning': False,
            'needs_execution': False,
            'query_type': 'general',
            'confidence_score': 0.5
        }
        
        try:
            # Execute the graph
            result = self.graph.invoke(state)
            
            execution_time = time.time() - start_time
            
            # Save AI response to chat history
            ai_response = result.get('output', '')
            self.chat_history.add_ai_message(ai_response)
            
            if self.verbose:
                logger.info(f"✅ Request processed successfully in {execution_time:.2f}s")
                logger.info(f"💾 Saved conversation to history (session: {self.session_id})")
            
            # Prepare result with plan_options if available
            response = {
                'output': result.get('output', ''),
                'agent_type': result.get('agent_type', 'direct'),
                'confidence': result.get('confidence_score', 0.5),
                'execution_time': execution_time,
                'success': True
            }
            
            # Include plan_options from delegation_result if available
            delegation_result = result.get('delegation_result', {})
            if delegation_result.get('plan_options'):
                response['plan_options'] = delegation_result['plan_options']
                
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Error in Manager Agent execution: {str(e)}")
            
            # Save error response to chat history
            error_message = f'I apologize, but I encountered an error while processing your request: {str(e)}'
            self.chat_history.add_ai_message(error_message)
            
            return {
                'output': f'I apologize, but I encountered an error while processing your request: {str(e)}',
                'agent_type': 'error',
                'confidence': 0.0,
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    def stream(self, input_data: str):
        """
        Streaming interface (placeholder for future implementation)
        """
        # For now, just return the regular invoke result
        return self.invoke(input_data)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of the Manager Agent and sub-agents
        """
        return {
            'manager_status': 'active',
            'sub_agents': {
                'plan_agent': self._plan_agent is not None,
                'meta_agent': self._meta_agent is not None,
                'tool_agent': self._tool_agent is not None
            },
            'cached_plans': len(self._cached_plan_options) > 0,
            'capabilities': get_agent_capabilities()
        }
    
    def clear_cache(self):
        """Clear cached plan options"""
        self._cached_plan_options = {}
        if self.verbose:
            logger.info("🗑️ Plan cache cleared")


# Export the ManagerAgent
__all__ = ["ManagerAgent"]