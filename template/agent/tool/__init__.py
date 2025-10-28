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

logger = logging.getLogger(__name__)

class ToolAgent:
    """Tool Agent sử dụng MCP để thực hiện smart home automation tasks"""

    def __init__(self, model="gemini-2.5-pro", temperature=0.2, verbose=False):
        self.name = "Tool Agent"
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.tools = []
        self.llm = None

    # ==========================================================
    # Init MCP tools + Vertex LLM
    # ==========================================================
    async def init_async(self):
        """Load MCP tools and initialize Vertex LLM"""
        self.tools = await self.get_mcp_tools()

        if self.verbose:
            logger.info(f"🔧 Loaded {len(self.tools)} MCP tools")
            for t in self.tools:
                logger.info(f"🔹 {t.name} - {getattr(t, 'description', '')}")

    async def get_mcp_tools(self):
        """Get tool list from MCP server"""
        try:
            async with MultiServerMCPClient(
                {"mcp-server": {"url": env.MCP_SERVER_URL, "transport": "sse"}}
            ) as client:
                return list(client.get_tools())
        except Exception as e:
            logger.error(f"❌ MCP connection failed: {e}")
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
            # For now, return a placeholder response
            # In a full implementation, this would use AutoGen with MCP tools
            return {
                "input": query,
                "token": token,
                "route": "execute",
                "output": f"Tool execution placeholder for: {query}",
                "error": "",
                "tool_data": {},
                "tool_agent_result": True  # Assume success for now
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