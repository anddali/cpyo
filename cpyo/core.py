

from abc import ABC, abstractmethod
import inspect
from typing import Callable, Generator, List, Dict, Any, Optional, get_type_hints
import json
from .event import AgentEvent, AgentEventType
from .memory import Message, Messages

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

    def _stream_final_response(self, messages, **kwargs) -> Generator[AgentEvent, None, None]:
        """Stream the final response token by token.
        
        Args:
            messages: The current message context
            **kwargs: Additional arguments for the provider
            
        Yields:
            AgentEvent: Events containing partial response tokens and the final complete response
        """
        # Make sure the streaming parameter is enabled for the provider
        kwargs["stream"] = True
        
        response_stream = self.provider.generate(
            messages.get_messages(),
            # tools=self.tool_schemas,
            # tool_choice="none",
            **kwargs
        )
        
        full_response = ""
        
        # Different providers have different streaming implementations
        # This is a generic approach that should work with OpenAI and similar providers
        for chunk in response_stream:
            if hasattr(chunk, "choices") and chunk.choices:
                # Extract delta content if available
                if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                    token = chunk.choices[0].delta.content
                    if token:
                        full_response += token
                        yield AgentEvent(
                            AgentEventType.PARTIAL_RESPONSE,
                            data={"token": token, "accumulated": full_response}
                        )
                # Some providers might provide different formats
                elif hasattr(chunk.choices[0], "text"):
                    token = chunk.choices[0].text
                    if token:
                        full_response += token
                        yield AgentEvent(
                            AgentEventType.PARTIAL_RESPONSE,
                            data={"token": token, "accumulated": full_response}
                        )
        
        # After streaming is complete, send the final complete response
        yield AgentEvent(
            AgentEventType.FINAL_RESPONSE,
            message="Task complete",
            data={"response": full_response}
        )

    def run(self, **kwargs) -> Generator[AgentEvent, None, None]:
        """Run the agent.
            
        Returns:
            Generator[AgentEvent, None, None]: A generator that yields events during the agent's execution
        """
        # This is a default implementation that subclasses should override
        raise NotImplementedError("Agent subclasses must implement their own run method.")


class ReActAgent(Agent):
    """Agent that uses a combination of reasoning and action with improved progress reporting."""
    def __init__(self, name: str, description: str, provider: LLMProvider, tools: List[Tool]):
        super().__init__(name, description, provider, tools)
        self._token_buffer = []  # Buffer for collecting tokens when streaming

    def run(self, **kwargs) -> Generator[AgentEvent, None, None]:
        """Run the ReAct agent with the given input.
        
        Args:
            **kwargs: Additional arguments including:
                - messages: List of conversation messages
                - stream: Boolean indicating whether to stream the final response
                
        Yields:
            AgentEvent: Structured event objects providing updates on agent progress
        """
        stream = kwargs.pop("stream", False)
        messages = kwargs.pop("messages", None)
        
        if messages is None or not isinstance(messages, Messages):
            yield AgentEvent(AgentEventType.ERROR, message="Invalid messages provided.")
            return
            
        # Setup reasoning
        reasoning_messages = messages.copy()
        reasoning_messages.add_system_message(
            """Do not answer the question yet. Analyze the question and think about how to approach it. What is user asking?
            Think step by step what do you need to answer the question.
            What is the problem? What are the subproblems?
            What information do you need? What tools might be helpful from the ones that you have access to? 
            If multiple tools are needed, think about the sequence in which they should be used and how data might flow between them.                        
            """) #If no tools are needed, at the end of your reasoning, print \{tools_needed=False\}
        
        yield AgentEvent(
            AgentEventType.THINKING, 
            message="Developing reasoning strategy..."
        )
              
        reasoning_response = self.provider.generate(
            reasoning_messages.get_messages(), 
            tools=self.tool_schemas, 
            tool_choice="none", 
            **kwargs
        )
        reasoning = self.provider.extract_content(reasoning_response)
        
        yield AgentEvent(
            AgentEventType.THINKING, 
            message="Reasoning complete", 
            data={"reasoning": reasoning}
        )

        # Setup action phase
        current_messages = messages.copy()
        current_messages.add_system_message(
            f"""Based on your reasoning: "{reasoning}"
                                            
            Now take appropriate action to solve the problem. You can use multiple tools in sequence if needed.
            If you need to use multiple tools, execute them one at a time and use the results from previous tools to inform subsequent tool calls.            
            Take only one action at a time, wait for the results, then decide on the next action based on those results."""
        )
        
        max_iterations = 5  # Limit to prevent infinite loops
        iteration = 0
        final_output = None

        while iteration < max_iterations:
            iteration += 1
            yield AgentEvent(
                AgentEventType.PROGRESS, 
                message=f"Starting iteration {iteration}",
                iteration=iteration
            )
            
            # Generate the next action based on the reasoning and current messages
            action_response = self.provider.generate(
                current_messages.get_messages(), 
                tools=self.tool_schemas, 
                tool_choice="auto", 
                **kwargs
            )
            content = self.provider.extract_content(action_response)
            tool_calls = self.provider.extract_tool_calls(action_response)

            yield AgentEvent(
                AgentEventType.THINKING, 
                message="Action generated", 
                data={"action": content}
            )


            if not tool_calls:
                if stream:
                    # If we should stream the final response, process it differently
                    yield from self._stream_final_response(current_messages, **kwargs)
                else:
                    yield AgentEvent(
                        AgentEventType.FINAL_RESPONSE, 
                        message="Task complete", 
                        data={"response": content}
                    )

                
                print("Current messages:")
                for message in current_messages.get_messages():
                    print("-----------------------------------------------------------------------------")
                    for key, value in message.items():                    
                        print(f"  {key}: {value}")
                return
            
            yield AgentEvent(
                AgentEventType.PROGRESS, 
                message=f"Found {len(tool_calls)} tool calls",
                iteration=iteration,
                data={"tool_count": len(tool_calls)}
            )

            # Execute tools and collect responses
            assistant_message = {"role": "assistant", "content": content}
            assistant_message["tool_calls"] = action_response.choices[0].message.tool_calls
            current_messages.add_message(Message(**assistant_message))
            
            # Process each tool call in this iteration
            for tool_idx, tool_call in enumerate(tool_calls):
                try:
                    if hasattr(tool_call, "function"):
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id
                    else:
                        function_name = tool_call["function"]["name"]
                        function_args = json.loads(tool_call["function"]["arguments"])
                        tool_call_id = tool_call["id"]
    
                    yield AgentEvent(
                        AgentEventType.TOOL_CALL, 
                        message=f"Calling tool: {function_name}",
                        iteration=iteration,
                        data={
                            "tool_name": function_name, 
                            "tool_args": function_args,
                            "tool_index": tool_idx,
                            "tool_call_id": tool_call_id
                        }
                    )
                    
                    # Execute the tool
                    if function_name in self.tool_map:
                        if self.tool_map[function_name].tool_type == ToolType.FUNCTION:
                            result = self.tool_map.get(function_name).run(**function_args)
                        elif self.tool_map[function_name].tool_type == ToolType.AGENT:
                            for event in self.tool_map[function_name].run(**kwargs):
                                if event.event_type == AgentEventType.FINAL_RESPONSE:
                                    result = event.data["response"]
                                    break
                                elif event.event_type == AgentEventType.ERROR:
                                    result = {"error": event.message}
                                    break
                    else:
                        result = {"error": f"Tool {function_name} not found."}
                        
                    yield AgentEvent(
                        AgentEventType.TOOL_RESULT,
                        message=f"Tool {function_name} execution complete",
                        iteration=iteration,
                        data={
                            "tool_name": function_name,
                            "result": result,
                            "tool_index": tool_idx,
                            "tool_call_id": tool_call_id
                        }
                    )
    
                    tool_response = self.provider.create_tool_response_message(
                        tool_call_id,
                        function_name,
                        json.dumps(result)
                    )
                    current_messages.add_message(Message(**tool_response))
                    
                except Exception as e:
                    error_msg = str(e)
                    yield AgentEvent(
                        AgentEventType.ERROR,
                        message=f"Error executing tool {function_name if 'function_name' in locals() else 'unknown'}",
                        iteration=iteration,
                        data={"error": error_msg}
                    )
                    
                    tool_response = self.provider.create_tool_response_message(
                        tool_call_id if 'tool_call_id' in locals() else "unknown",
                        function_name if 'function_name' in locals() else "unknown",
                        json.dumps({"error": error_msg})
                    )
                    current_messages.add_message(Message(**tool_response))
            
            # After tools execution, ask if we need more tools or if we have the final answer
            current_messages.add_system_message(
                """Based on the tool results above, decide if:
                1) You need to make additional tool calls to complete the task (if so, make the next appropriate tool call), or
                2) You have all the information needed to provide a final answer (if so, provide your comprehensive final response with no tool calls).
                
                Make sure to review all previous tool results when making your decision.
                Make sure to that the final answer is conversational and easy to understand.""")
               
        
        # If we've reached max iterations without a final answer, synthesize from what we have
        if iteration >= max_iterations:
            yield AgentEvent(
                AgentEventType.PROGRESS,
                message=f"Reached maximum iterations ({max_iterations}). Synthesizing final answer...",
                iteration=iteration
            )
            
            if stream:
                yield from self._stream_final_response(current_messages, **kwargs)
            else:
                final_response = self.provider.generate(
                    current_messages.get_messages(), 
                    tools=self.tool_schemas, 
                    tool_choice="none", 
                    **kwargs
                )
                final_output = self.provider.extract_content(final_response)
                
                yield AgentEvent(
                    AgentEventType.FINAL_RESPONSE,
                    message="Task complete (max iterations reached)",
                    data={"response": final_output}
                )
                            
            print("Current messages:")
            for message in current_messages.get_messages():
                print("-----------------------------------------------------------------------------")
                for key, value in message.items():                    
                    print(f"  {key}: {value}")
                    

    

