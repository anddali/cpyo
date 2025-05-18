from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Union
import json

class AgentEventType(Enum):
    """Enum to define different types of events that can be yielded by the agent."""
    THINKING = "thinking"        # Agent is reasoning/thinking
    TOOL_CALL = "tool_call"      # Agent is calling a tool
    TOOL_RESULT = "tool_result"  # Result from a tool call
    PROGRESS = "progress"        # General progress update
    ERROR = "error"              # Error occurred
    PARTIAL_RESPONSE = "partial" # Part of streaming final response
    FINAL_RESPONSE = "final"     # Final complete response

class AgentEvent:
    """Structured event object yielded by the agent during execution."""
    def __init__(
        self,
        event_type: AgentEventType,
        data: Dict[str, Any] = None,
        message: str = None,
        iteration: int = None
    ):
        self.event_type = event_type
        self.data = data or {}
        self.message = message
        self.iteration = iteration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = {
            "type": self.event_type.value,
            "message": self.message,
        }
        if self.data:
            result["data"] = self.data
        if self.iteration is not None:
            result["iteration"] = self.iteration
        return result