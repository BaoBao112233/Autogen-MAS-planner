"""
Message converter utility to convert between custom message types and LangChain message types
"""
from langchain_core.messages import AIMessage as LangChainAIMessage
from langchain_core.messages import HumanMessage as LangChainHumanMessage
from langchain_core.messages import SystemMessage as LangChainSystemMessage

from template.message.message import (
    AIMessage as CustomAIMessage,
    HumanMessage as CustomHumanMessage, 
    SystemMessage as CustomSystemMessage,
    BaseMessage
)

def convert_to_langchain_message(message):
    """
    Convert custom message types to LangChain message types
    """
    if isinstance(message, CustomSystemMessage):
        return LangChainSystemMessage(content=message.content)
    elif isinstance(message, CustomHumanMessage):
        return LangChainHumanMessage(content=message.content)
    elif isinstance(message, CustomAIMessage):
        return LangChainAIMessage(content=message.content)
    elif isinstance(message, (LangChainSystemMessage, LangChainHumanMessage, LangChainAIMessage)):
        # Already a LangChain message, return as-is
        return message
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

def convert_messages_list(messages):
    """
    Convert a list of messages to LangChain message types
    """
    return [convert_to_langchain_message(msg) for msg in messages]