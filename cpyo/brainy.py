

from abc import ABC, abstractmethod
import inspect
from typing import Callable, List, Dict, Any, Optional, get_type_hints

from cpyo.memory import Messages

from .llm_providers import LLMProvider

class ToolType:
    """Enum for tool types."""
    FUNCTION = "function"
    AGENT = "agent"
    

class Tool(ABC):
    """Abstract base class for all tools.

    This class is used to define the interface for all tools in the system.
    It provides a common structure and ensures that all tools implement the
    required methods.
    """
    def __init__(self, name: str, description: str, tool_type: ToolType, function: Callable = None):
        self.name = name
        self.description = description
        self.signature = None
        self.function = function
        self.tool_type = tool_type
        if function is not None:
            self.signature = self.__get_function_signature__(function)
        else:
            self.signature = self.__get_function_signature__(self.run)
    
    def __get_function_signature__(self, func: Optional[Callable] = None) -> Optional[inspect.Signature]:
        """Get the signature of the function."""
        if func is None:
            func = self.function
            
        if func is None:
            return None
            
        return inspect.signature(func)    

    def to_schema(self) -> dict:
        """Convert the tool to OpenAI tool schema format."""
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        if self.signature:
            self._populate_schema_parameters(schema, self.signature, self.function.__doc__ if self.function else self.run.__doc__)
        
        return schema
    
    def _populate_schema_parameters(self, schema: Dict[str, Any], signature: inspect.Signature, docstring: Optional[str] = None) -> None:
        """Populate schema parameters from a function signature and docstring."""
        for param_name, param in signature.parameters.items():
            # Skip self parameter for class methods
            if param_name == 'self':
                continue
                
            param_type = "string"  # Default type
            param_description = ""
            
            # Try to get type annotation and convert to JSON schema type
            if param.annotation != inspect.Parameter.empty:
                param_type = self._get_json_schema_type(param.annotation)
            
            # Extract description from function docstring if available
            if docstring:
                import re
                doc_lines = docstring.split('\n')
                for i, line in enumerate(doc_lines):
                    param_match = re.search(rf'{param_name}\s*\((.*?)\):\s*(.*)', line)
                    if param_match:
                        param_description = param_match.group(2).strip()
                        break
            
            schema["function"]["parameters"]["properties"][param_name] = {
                "type": param_type,
                "description": param_description
            }
            
            # Add to required list if the parameter doesn't have a default value
            if param.default == inspect.Parameter.empty:
                schema["function"]["parameters"]["required"].append(param_name)
    
    def _get_json_schema_type(self, annotation: Any) -> str:
        """Convert Python type annotation to JSON schema type."""
        if annotation == int:
            return "integer"
        elif annotation == float:
            return "number"
        elif annotation == bool:
            return "boolean"
        elif annotation == list or getattr(annotation, "__origin__", None) == list:
            return "array"
        elif annotation == dict or getattr(annotation, "__origin__", None) == dict:
            return "object"
        else:
            return "string"

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the tool with the given arguments."""
        raise NotImplementedError("Subclasses must implement this method.")


class FunctionTool(Tool):
    """Concrete implementation of Tool for functions."""
    def __init__(self, name: str, description: str, function: Callable):
        super().__init__(name, description, ToolType.FUNCTION, function)

    def run(self, *args, **kwargs):
        """Run the function with the given arguments."""
        if self.function is None:
            raise ValueError("Function is not defined.")
        return self.function(*args, **kwargs)


class PythonTool(Tool):
    """Tool for executing Python code given as a text input."""
    def __init__(self, name: str, description: str):
        super().__init__(name, description, ToolType.FUNCTION)    

    def run(self, code: str) -> Any:
        """Execute the given Python code.

            Args:
                code (str): The Python code to execute.

            Returns:
                Any: The result of the executed code.
        """
        try:
            # Execute the code and return the result
            local_scope = {}
            exec(code, {}, local_scope)
            if "result" in local_scope:
                return local_scope["result"]
            else:
                raise RuntimeError("No result variable found in the executed code.")    
        except Exception as e:
            raise RuntimeError(f"Error executing code: {e}")


class Agent(Tool):
    """Base class for agents that can use multiple tools."""
    def __init__(self, name: str, description: str, provider:LLMProvider, tools: List[Tool]):
        super().__init__(name, description, ToolType.AGENT, None)
        self.tools = tools
        self.provider = provider        
        if tools is None or len(tools) == 0:
            self.tool_map = {}
            self.tool_schemas = []
        else:
            self.tool_map = {tool.name: tool for tool in tools}
            self.tool_schemas = [tool.to_schema() for tool in tools]
        
        

    def run(self, input: str, **kwargs) -> str:
        """Run the agent with the given input.
        
        Args:
            input (str): The user input to process.
            
        Returns:
            str: The agent's response.
        """
        # This is a default implementation that subclasses should override
        raise NotImplementedError("Agent subclasses must implement their own run method.")

class ReActAgent(Agent):
    """Agent that uses a combination of reasoning and action."""
    def __init__(self, name: str, description: str, provider: LLMProvider, tools: List[Tool]):
        super().__init__(name, description, provider, tools)
        

    def run(self, **kwargs) -> Any: 
        """Run the ReAct agent with the given input."""
        yield {"status": "update", "message": "Starting IterativeReasonActAgent..."}
        messages = kwargs.pop("messages", None)
        if messages is None:
            raise ValueError("Messages are required for ReAct agent.")
        reasoning_messages = messages.copy()
        reasoning_messages.add_system_message(            
            """Think step by step about how to solve this problem. 
            What information do you need? What tools might be helpful? 
            If multiple tools are needed, think about the sequence in which they should be used and how data might flow between them.
            Answers should be based primarily on tools responses even if they are contradictory, only when not found, use your own knowledge. 
            Responses should always reference where info comes from if the links if available.
            If no tools are needed, at the end of your reasoning, print \{tools_needed=False\}""")
        
        yield {"status": "update", "message": "Developing reasoning strategy..."}        
        reasoning_response = self.provider.generate(reasoning_messages.get_messages(), tools=self.tool_schemas, tool_choice="none", **kwargs)
        reasoning = self.provider.extract_content(reasoning_response)
        yield {"status": "update", "message": "Reasoning complete: " + reasoning}

        current_messages = messages.copy()
        current_messages.add_system_message("""Based on your reasoning: {reasoning}            
                                            
            Now take appropriate action to solve the problem. You can use multiple tools in sequence if needed.
            If you need to use multiple tools, execute them one at a time and use the results from previous tools to inform subsequent tool calls.
            Any time you need to answer knowledge questions, use the the enterprise search tool.
            Take only one action at a time, wait for the results, then decide on the next action based on those results.""")
        
        max_iterations = 5  # Limit to prevent infinite loops
        iteration = 0
        final_output = None

        while iteration < max_iterations:
            iteration += 1
            yield {"status": "update", "message": f"Iteration {iteration}..."}
            
            # Generate the next action based on the reasoning and current messages
            action_response = self.provider.generate(current_messages.get_messages(), tools=self.tool_schemas, tool_choice="auto", **kwargs)
            content = self.provider.extract_content(action_response)
            tool_calls = self.provider.extract_tool_calls(action_response)
            
            if not tool_calls:
                    yield {"status": "update", "message": "Finalizing response..."}
                    final_output = content
                    break
            
            yield {"status": "update", "message": f"Found {len(tool_calls)} tool calls in iteration {iteration}."}

    