"""
Tool Agent for MAS-Planning system using MCP tools
Converted to use AutoGen instead of LangChain
"""
import logging
import asyncio
import json
from typing import Dict, Any, List
from termcolor import colored
from template.configs.environments import env
from template.message.message import HumanMessage, SystemMessage
from template.message.converter import convert_messages_list
from template.agent.tool.prompt import TOOL_PROMPT
from autogen import AssistantAgent, UserProxyAgent
from langchain_mcp_adapters.client import MultiServerMCPClient
from template.agent.autogen_config import create_autogen_agent, create_user_proxy_agent

logger = logging.getLogger(__name__)

class ToolAgent:
    """Tool Agent sử dụng MCP để thực hiện smart home automation tasks"""

    def __init__(self, model="gemini-2.5-pro", temperature=0.2, verbose=False):
        self.name = "Tool Agent"
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.tools = []
        self.agent = None

    # ==========================================================
    # Init MCP tools + Vertex LLM
    # ==========================================================
    async def init_async(self):
        """Load MCP tools and initialize Vertex LLM"""
        self.tools = await self.get_mcp_tools()

        # Create AutoGen agent with MCP tools
        self.agent = create_autogen_agent(
            name=self.name,
            system_message=TOOL_PROMPT,
            model=self.model,
            temperature=self.temperature,
            tools=self.tools
        )

        if self.verbose:
            logger.info(f"🔧 Loaded {len(self.tools)} MCP tools")
            for t in self.tools:
                logger.info(f"🔹 {t.name} - {getattr(t, 'description', '')}")

    async def get_mcp_tools(self):
        """Get tool list from MCP server"""
        try:
            logger.info(f"🔗 Attempting to connect to MCP server at {env.MCP_SERVER_URL}")
            async with MultiServerMCPClient(
                {"mcp-server": {"url": env.MCP_SERVER_URL, "transport": "sse"}}
            ) as client:
                tools = list(client.get_tools())
                logger.info(f"✅ Successfully connected to MCP server, loaded {len(tools)} tools")
                return tools
        except Exception as e:
            logger.warning(f"⚠️  MCP connection failed: {e}. Continuing without MCP tools.")
            logger.info("💡 To enable MCP tools, ensure MCP server is running on the configured URL")
            return []

    # ==========================================================
    # Public interface
    # ==========================================================
    def invoke(self, input_data, **kwargs):
        """Main invoke method"""
        token = kwargs.get("token", "")
        query = input_data["input"] if isinstance(input_data, dict) else input_data

        if self.verbose:
            logger.info(f"🔧 ToolAgent processing: {query}")

        try:
            if not self.agent:
                # Initialize agent if not done yet
                asyncio.run(self.init_async())

            if not self.tools:
                # No MCP tools available, return placeholder
                return {
                    "input": query,
                    "token": token,
                    "route": "execute",
                    "output": f"Tool execution placeholder (no MCP tools available): {query}",
                    "error": "",
                    "tool_data": {},
                    "tool_agent_result": True
                }

            # Create user proxy for this interaction
            user_proxy = create_user_proxy_agent(
                name="tool_user_proxy",
                code_execution_config=False,
                human_input_mode="NEVER"
            )

            # Prepare system message with available tools
            tool_names = [t.name for t in self.tools]
            system_with_tools = TOOL_PROMPT + f"\n\nAvailable MCP Tools: {', '.join(tool_names)}"

            # Update agent system message
            self.agent.update_system_message(system_with_tools)

            # Initiate chat with the agent
            chat_result = user_proxy.initiate_chat(
                self.agent,
                message=f"Execute this smart home automation task: {query}",
                max_turns=3  # Allow for tool calls and responses
            )

            # Extract the final response
            if chat_result and chat_result.chat_history:
                final_response = chat_result.chat_history[-1].get('content', '')

                return {
                    "input": query,
                    "token": token,
                    "route": "execute",
                    "output": final_response,
                    "error": "",
                    "tool_data": {},
                    "tool_agent_result": True
                }
            else:
                return {
                    "input": query,
                    "token": token,
                    "route": "execute",
                    "output": "No response from tool execution",
                    "error": "",
                    "tool_data": {},
                    "tool_agent_result": False
                }

        except Exception as e:
            logger.error(f"❌ ToolAgent error: {str(e)}")
            return {
                "input": query,
                "token": token,
                "route": "error",
                "output": f"Tool execution failed: {str(e)}",
                "error": str(e),
                "tool_data": {},
                "tool_agent_result": False
            }

    def stream(self, input: str):
        """Placeholder for streaming"""
        pass


__all__ = ["ToolAgent"]