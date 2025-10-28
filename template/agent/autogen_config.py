"""
AutoGen LLM Configuration Utilities

This module provides utilities for configuring AutoGen agents with Vertex AI
and other LLM providers, replacing LangChain integrations.
"""

import os
from typing import Dict, Any, Optional
from autogen import AssistantAgent
from template.configs.environments import env


def create_vertex_llm_config(
    model: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create AutoGen LLM configuration for Vertex AI models.

    Args:
        model: The model name (e.g., "gemini-2.5-pro", "gemini-1.5-pro")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional configuration options

    Returns:
        Dict containing AutoGen LLM configuration
    """
    config = {
        "model": model,
        "api_type": "google",
        "model_name": model,
        "temperature": temperature,
        "project_id": env.GOOGLE_CLOUD_PROJECT,
        "location": env.GOOGLE_CLOUD_LOCATION,
    }

    if max_tokens:
        config["max_tokens"] = max_tokens

    # Add any additional kwargs
    config.update(kwargs)

    return config


def create_autogen_agent(
    name: str,
    system_message: str,
    model: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    tools: Optional[list] = None,
    **kwargs
) -> AssistantAgent:
    """
    Create a configured AutoGen AssistantAgent.

    Args:
        name: Agent name
        system_message: System prompt for the agent
        model: Model name
        temperature: Sampling temperature
        tools: List of tools/functions available to the agent
        **kwargs: Additional agent configuration

    Returns:
        Configured AssistantAgent
    """
    llm_config = create_vertex_llm_config(model=model, temperature=temperature)

    # Add tools to config if provided
    if tools:
        llm_config["tools"] = tools
        llm_config["tool_choice"] = "auto"  # Enable automatic tool selection

    agent_config = {
        "name": name,
        "system_message": system_message,
        "llm_config": llm_config,
    }

    # Add any additional kwargs
    agent_config.update(kwargs)

    return AssistantAgent(**agent_config)


def create_user_proxy_agent(
    name: str = "user_proxy",
    code_execution_config: bool = False,
    human_input_mode: str = "NEVER",
    **kwargs
):
    """
    Create a configured AutoGen UserProxyAgent.

    Args:
        name: Agent name
        code_execution_config: Whether to enable code execution
        human_input_mode: Human input mode ("ALWAYS", "TERMINATE", "NEVER")
        **kwargs: Additional agent configuration

    Returns:
        Configured UserProxyAgent
    """
    from autogen import UserProxyAgent

    config = {
        "name": name,
        "code_execution_config": code_execution_config,
        "human_input_mode": human_input_mode,
    }

    config.update(kwargs)

    return UserProxyAgent(**config)