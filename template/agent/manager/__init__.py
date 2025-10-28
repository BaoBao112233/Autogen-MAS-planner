"""
Manager Agent - Central coordinator for the Multi-Agent System (MAS)

The Manager Agent serves as the entry point and orchestrator for the entire MAS system.
It analyzes user queries, makes routing decisions, and coordinates interactions between
specialized agents (Plan Agent, Meta Agent, Tool Agent).
"""

from template.agent import BaseAgent
from template.configs.environments import env
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
from template.agent.autogen_config import create_autogen_agent, create_user_proxy_agent, create_vertex_llm_config

from autogen import GroupChat, GroupChatManager
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

    Uses AutoGen GroupChat for multi-agent orchestration instead of LangGraph
    """

    def __init__(self,
                 session_id: str,
                 conversation_id: str,
                 model: str = "gemini-2.5-pro",
                 temperature: float = 0.2,
                 max_iteration: int = 5,
                 verbose: bool = False):
        super().__init__()

        self.name = "Manager Agent"

        # Chat history for conversation context
        self.session_id = session_id
        self.chat_history = RedisSupportChatHistory(
            session_id=session_id,
            conversation_id=conversation_id,
            ttl=env.TTL_SECONDS
        )

        self.model = model
        self.temperature = temperature
        self.max_iteration = max_iteration
        self.verbose = verbose

        # Lazy-loaded sub-agents
        self._plan_agent = None
        self._meta_agent = None
        self._tool_agent = None

        # Session state for plan persistence
        self._cached_plan_options = {}

        # Create AutoGen GroupChat system
        self.group_chat = None
        self.group_chat_manager = None
        self._setup_group_chat()

        if self.verbose:
            logger.info(f"✅ {self.name} initialized successfully")
    
    def _setup_group_chat(self):
        """Setup AutoGen GroupChat with specialized agents"""
        # Create manager agent for coordination
        manager_agent = create_autogen_agent(
            name="Coordinator",
            system_message=MANAGER_PROMPT,
            model=self.model,
            temperature=self.temperature
        )

        # Create user proxy for initiating conversations
        user_proxy = create_user_proxy_agent(
            name="user_proxy",
            human_input_mode="NEVER"
        )

        # Initialize sub-agents
        self._ensure_sub_agents_loaded()

        # Create group chat with all agents
        agents = [user_proxy, manager_agent]

        # Add specialized agents if available
        if self._plan_agent and hasattr(self._plan_agent, 'agent'):
            agents.append(self._plan_agent.agent)
        if self._meta_agent and hasattr(self._meta_agent, 'agent'):
            agents.append(self._meta_agent.agent)
        if self._tool_agent and hasattr(self._tool_agent, 'agent'):
            agents.append(self._tool_agent.agent)

        # Create group chat
        self.group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=self.max_iteration,
            speaker_selection_method="round_robin"  # Let agents take turns
        )

        # Create group chat manager
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=create_vertex_llm_config(self.model, self.temperature)
        )

    def _ensure_sub_agents_loaded(self):
        """Ensure all sub-agents are loaded"""
        # Load Plan Agent
        if self._plan_agent is None:
            from template.agent.plan import PlanAgent
            self._plan_agent = PlanAgent(verbose=self.verbose)

        # Load Meta Agent
        if self._meta_agent is None:
            from template.agent.meta import MetaAgent
            self._meta_agent = MetaAgent(verbose=self.verbose)

        # Load Tool Agent
        if self._tool_agent is None:
            from template.agent.tool import ToolAgent
            self._tool_agent = ToolAgent(verbose=self.verbose)

    @property
    def plan_agent(self):
        """Lazy load Plan Agent"""
        if self._plan_agent is None:
            from template.agent.plan import PlanAgent
            self._plan_agent = PlanAgent(verbose=self.verbose)
        return self._plan_agent

    @property
    def meta_agent(self):
        """Lazy load Meta Agent"""
        if self._meta_agent is None:
            from template.agent.meta import MetaAgent
            self._meta_agent = MetaAgent(verbose=self.verbose)
        return self._meta_agent

    @property
    def tool_agent(self):
        """Lazy load Tool Agent"""
        if self._tool_agent is None:
            from template.agent.tool import ToolAgent
            self._tool_agent = ToolAgent(verbose=self.verbose)
        return self._tool_agent

    def _analyze_query_intent(self, user_input: str) -> str:
        """
        Analyze user query to determine intent using simple heuristics
        (Simplified version for AutoGen integration)
        """
        query_lower = user_input.lower().strip()

        # Priority 1: Plan creation
        if any(word in query_lower for word in ['create plan', 'plan', 'create', 'setup', 'automate']):
            return 'plan'

        # Priority 2: Plan selection
        if any(word in query_lower for word in ['plan 1', 'plan 2', 'plan 3', '1', '2', '3']):
            return 'plan'

        # Priority 3: Device control
        if any(word in query_lower for word in ['turn', 'set', 'control', 'adjust']):
            return 'tool'

        # Priority 4: Analysis requests
        if any(word in query_lower for word in ['analyze', 'analysis', 'evaluate', 'think']):
            return 'meta'

        # Default to direct response
        return 'direct'

    def invoke(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for the Manager Agent using AutoGen GroupChat

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

        try:
            # Analyze query intent
            intent = self._analyze_query_intent(user_message)

            if self.verbose:
                logger.info(f"🎯 Detected intent: {intent}")

            # Route to appropriate agent based on intent
            if intent == 'plan':
                # Handle plan-related queries
                result = self._handle_plan_request(input_data)
            elif intent == 'tool':
                # Handle tool/device control queries
                result = self._handle_tool_request(input_data)
            elif intent == 'meta':
                # Handle analysis queries
                result = self._handle_meta_request(input_data)
            else:
                # Handle direct queries with group chat
                result = self._handle_direct_request(user_message)

            execution_time = time.time() - start_time

            # Save AI response to chat history
            ai_response = result.get('output', '')
            self.chat_history.add_ai_message(ai_response)

            if self.verbose:
                logger.info(f"✅ Request processed successfully in {execution_time:.2f}s")

            # Prepare result
            response = {
                'output': result.get('output', ''),
                'agent_type': result.get('agent_type', 'direct'),
                'confidence': result.get('confidence', 0.5),
                'execution_time': execution_time,
                'success': True
            }

            # Include plan_options if available
            if 'plan_options' in result:
                response['plan_options'] = result['plan_options']

            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Error in Manager Agent execution: {str(e)}")

            # Save error response to chat history
            error_message = f'I apologize, but I encountered an error while processing your request: {str(e)}'
            self.chat_history.add_ai_message(error_message)

            return {
                'output': error_message,
                'agent_type': 'error',
                'confidence': 0.0,
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }

    def _handle_plan_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle plan-related requests"""
        user_message = input_data.get('message', '') or input_data.get('input', '')
        token = input_data.get('token', '')

        # Check if this is a plan selection
        plan_selection = extract_plan_selection(user_message)

        if plan_selection and self._cached_plan_options:
            # User is selecting from cached plans
            result = self.plan_agent.invoke(
                user_message,
                selected_plan_id=plan_selection,
                plan_options=self._cached_plan_options,
                token=token
            )
        else:
            # New plan creation request
            result = self.plan_agent.invoke(user_message, token=token)

            # Cache plan options for future selections
            if result.get('plan_options'):
                self._cached_plan_options = result['plan_options']

        return {
            'output': result.get('output', ''),
            'agent_type': 'plan',
            'plan_options': result.get('plan_options'),
            'confidence': 0.8
        }

    def _handle_tool_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool/device control requests"""
        token = input_data.get('token', '')
        tool_input = {
            'input': input_data.get('message', '') or input_data.get('input', ''),
            'token': token
        }

        result = self.tool_agent.invoke(tool_input)

        return {
            'output': result.get('output', ''),
            'agent_type': 'tool',
            'confidence': 0.8
        }

    def _handle_meta_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis requests"""
        user_message = input_data.get('message', '') or input_data.get('input', '')

        result = self.meta_agent.invoke(user_message)

        return {
            'output': result.get('output', ''),
            'agent_type': 'meta',
            'confidence': 0.7
        }

    def _handle_direct_request(self, user_message: str) -> Dict[str, Any]:
        """Handle direct requests using group chat"""
        try:
            # Use the group chat manager to coordinate response
            chat_result = self.group_chat_manager.initiate_chat(
                recipient=self.group_chat_manager,
                message=user_message,
                max_turns=3
            )

            # Extract final response
            if chat_result and chat_result.chat_history:
                final_response = chat_result.chat_history[-1].get('content', '')
            else:
                final_response = """🤖 **Smart Home Assistant**

I'm your Multi-Agent Smart Home Assistant! I can help you with:

🏗️ **Planning**: Create customized smart home automation plans
🔧 **Control**: Manage and control your smart devices
🧠 **Analysis**: Provide strategic insights for home automation

**Popular Commands:**
• "Create a smart home plan for my bedroom"
• "Turn on the living room lights"
• "Set the air conditioner to 22 degrees"

How can I assist you today?"""

            return {
                'output': final_response,
                'agent_type': 'direct',
                'confidence': 0.6
            }

        except Exception as e:
            logger.error(f"❌ Group chat error: {str(e)}")

            # Fallback response
            return {
                'output': "I apologize, but I'm having trouble processing your request right now. Please try again.",
                'agent_type': 'direct',
                'confidence': 0.3
            }

    def stream(self, input_data: str):
        """
        Streaming interface (placeholder for future implementation)
        """
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
            
# Export the ManagerAgent
__all__ = ["ManagerAgent"]