from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    @abstractmethod
    def invoke(self, input_data: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def stream(self, input_data: str):
        pass