import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from autogen import AssistantAgent, UserProxyAgent
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
import vertexai

from template.configs.environments import env

# Set environment variables for Google Cloud authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(env.GOOGLE_APPLICATION_CREDENTIALS)
os.environ["GOOGLE_CLOUD_PROJECT"] = env.GOOGLE_CLOUD_PROJECT

# Initialize Vertex AI
vertexai.init(project=env.GOOGLE_CLOUD_PROJECT, location=env.GOOGLE_CLOUD_LOCATION)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    @abstractmethod
    def invoke(self, input_data: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def stream(self, input_data: str):
        pass

class AutoGenAgent(BaseAgent):
    """Base class for AutoGen-based agents"""

    def __init__(self, name: str, system_message: str, model: str = "gemini-2.5-pro", temperature: float = 0.2):
        self.name = name
        self.model = model
        self.temperature = temperature

        # Create AutoGen AssistantAgent with Vertex AI LLM
        self.agent = AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config={
                "config_list": [{
                    "model": model,
                    "api_type": "vertexai",
                    "temperature": temperature,
                    "project_id": env.GOOGLE_CLOUD_PROJECT,
                    "location": env.GOOGLE_CLOUD_LOCATION,
                }],
            },
        )

        logger.info(f"Initialized AutoGen AssistantAgent: {name} with model {model}")

    def invoke(self, input_data: Any) -> Dict[str, Any]:
        """Default invoke method - should be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement invoke method")

    def stream(self, input_data: str):
        """Default stream method - placeholder"""
        pass

# Utility function to create Vertex AI LLM config for AutoGen
def create_vertex_llm_config(model: str = "gemini-2.5-pro", temperature: float = 0.2):
    return {
        "config_list": [{
            "model": model,
            "api_type": "vertexai",
            "temperature": temperature,
            "project_id": env.GOOGLE_CLOUD_PROJECT,
            "location": env.GOOGLE_CLOUD_LOCATION,
        }],
    }