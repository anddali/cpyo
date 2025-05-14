import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMProvider(ABC):
    """Abstract base class for different LLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None, **kwargs) -> Any:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from the LLM response."""
        pass
    
    @abstractmethod
    def extract_content(self, response: Any) -> str:
        """Extract content from the LLM response."""
        pass
    
    @abstractmethod
    def create_tool_response_message(self, tool_call_id: str, name: str, content: str) -> Dict[str, Any]:
        """Create a message with tool response."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLM provider."""
    
    def __init__(self, api_key=None):
        import openai
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None, **kwargs) -> Any:
        """Generate a response using OpenAI API."""
        model = kwargs.get("model", "gpt-4.1-nano")
        stream = kwargs.get("stream", False)        
        
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,            
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
            
        return self.client.chat.completions.create(**params)
    
    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from OpenAI response."""
        message = response.choices[0].message
        return message.tool_calls if hasattr(message, "tool_calls") and message.tool_calls else []
    
    def extract_content(self, response: Any) -> str:
        """Extract content from OpenAI response."""
        return response.choices[0].message.content or ""
    
    def create_tool_response_message(self, tool_call_id: str, name: str, content: str) -> Dict[str, Any]:
        """Create an OpenAI message with tool response."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content
        }


