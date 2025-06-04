from abc import ABC, abstractmethod
import inspect
from typing import Callable, Generator, List, Dict, Any, Optional, get_type_hints
import json
from .event import AgentEvent, AgentEventType
from .messages import Message, Messages
import datetime
from .llm_providers import LLMProvider
import datetime
import os
import re
from string import Template


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

    def _extract_description_from_docstring(self, docstring: str) -> str:
        """Extract only the description part from docstring, excluding Args section."""
        if not docstring:
            return self.description
        
        lines = docstring.strip().split('\n')
        description_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            # Stop when we hit Args: section
            if stripped_line.lower().startswith('args:'):
                break
            description_lines.append(stripped_line)
        
        # Join lines and clean up extra whitespace
        description = ' '.join(description_lines).strip()
        return description if description else self.description

    def to_schema(self) -> dict:
        """Convert the tool to OpenAI tool schema format."""
        docstring = self.function.__doc__ if self.function else self.run.__doc__
        description = self._extract_description_from_docstring(docstring)
        
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        if self.signature:
            self._populate_schema_parameters(schema, self.signature, docstring)
        
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
                # Look for parameter descriptions in Args section
                args_section_match = re.search(r'Args:\s*\n(.*?)(?=\n\s*(?:Returns|Raises|Note|Example|$))', docstring, re.DOTALL | re.IGNORECASE)
                if args_section_match:
                    args_content = args_section_match.group(1)
                    # Match parameter name followed by optional type in parentheses and description
                    param_pattern = rf'\s*{re.escape(param_name)}\s*(?:\([^)]*\))?\s*:\s*(.+?)(?=\n\s*\w+\s*(?:\([^)]*\))?\s*:|$)'
                    param_match = re.search(param_pattern, args_content, re.DOTALL)
                    if param_match:
                        param_description = param_match.group(1).strip().replace('\n', ' ')
            
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
        super().__init__(name, description, ToolType.FUNCTION);  

    def run(self, code: str) -> Any:
        """Tool for executing Python code represented as a single-line string. It is intended for performing internal operations within an agent, such as performing calculations and should never be used to execute user supplied code.
        
        Args:
            code (str): A **single-line string** of valid Python code to execute.
                - The code must assign the final result to a variable named `result`.
                - The code must be formatted as a **single line**, with statements separated by semicolons (`;`) if needed.                
                - All necessary imports must be included in the same line if used.
                - Do not use multi-line strings or triple quotes for the code.
                - Do not use slashes (`\`) to break lines.
                
                ✅ Valid example:
                >>> code = "import math; result = math.sqrt(16)"
                
                ❌ Invalid example (multi-line):
                >>> code = '''
                ... import math
                ... result = math.sqrt(16)
                ... '''

        Returns:
            Any: The value assigned to the `result` variable after execution.

        Raises:
            RuntimeError: If the code does not define a `result` variable or if any error occurs during execution.
        """
        try:
            # Execute the code and return the result
            local_scope = {}
            import math, random, datetime, json
            safe_globals = {
                "__builtins__": __builtins__,
                "math": math,
                "random": random,
                "datetime": datetime,
                "json": json,
            }
            exec(code, safe_globals, local_scope)            
            if "result" in local_scope:
                return local_scope["result"]
            else:
                raise RuntimeError("No result variable found in the executed code.")    
        except Exception as e:
            raise RuntimeError(f"Error executing code: {e}")

def tool(func=None, *, name: str = None, description: str = None):
    """Decorator to convert a function into a FunctionTool."""
    def decorator(f):
        tool_name = name or f.__name__
        tool_description = description or f.__doc__ or f"Tool: {f.__name__}"
        
        function_tool = FunctionTool(
            name=tool_name,
            description=tool_description,
            function=f
        )
        
        function_tool._original_function = f
        return function_tool
    
    if func is None:
        # Called with arguments: @tool(name="...", description="...")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)

class ApiTool(Tool):
    """Tool for making HTTP API requests with configurable parameters."""
    
    def __init__(
        self,
        name: str,
        description: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        body_template: Optional[str] = None,
        query_params_template: Optional[Dict[str, str]] = None,
        parameter_descriptions: Optional[Dict[str, str]] = None
    ):
        super().__init__(name, description, ToolType.FUNCTION, None)
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}
        self.auth = auth
        self.timeout = timeout
        self.body_template = body_template
        self.query_params_template = query_params_template or {}
        self.parameter_descriptions = parameter_descriptions or {}
    
    def _extract_template_variables(self, template_str: str) -> List[str]:
        """Extract variable names from a template string."""
        if not template_str:
            return []
        
        # Find all {var} style placeholders - improved regex to handle JSON properly
        # This pattern looks for { followed by word characters, then }
        pattern = r'\{(\w+)\}'
        matches = re.findall(pattern, template_str)
        variables = []
        for var_name in matches:
            # Clean up the variable name and ensure it's valid
            var_name = var_name.strip()
            if var_name and var_name not in variables:
                variables.append(var_name)
        return variables
    
    def _get_all_template_variables(self) -> Dict[str, str]:
        """Get all template variables from URL, body, query params, headers, and auth."""
        variables = {}
        
        # Extract from URL
        url_vars = self._extract_template_variables(self.url)
        for var in url_vars:
            # Use custom description if provided, otherwise use default
            variables[var] = self.parameter_descriptions.get(var, "URL path parameter")
        
        # Extract from body template
        if self.body_template:
            body_vars = self._extract_template_variables(self.body_template)
            for var in body_vars:
                # Use custom description if provided, otherwise use default
                variables[var] = self.parameter_descriptions.get(var, "Request body parameter")
        
        # Extract from query parameters
        for key, value in self.query_params_template.items():
            if isinstance(value, str):
                query_vars = self._extract_template_variables(value)
                for var in query_vars:
                    # Use custom description if provided, otherwise use default
                    variables[var] = self.parameter_descriptions.get(var, f"Query parameter for '{key}'")
        
        # Extract from headers
        for key, value in self.headers.items():
            if isinstance(value, str):
                header_vars = self._extract_template_variables(value)
                for var in header_vars:
                    # Use custom description if provided, otherwise use default
                    variables[var] = self.parameter_descriptions.get(var, f"Header parameter for '{key}'")
        
        # Extract from auth
        if self.auth:
            auth_vars = []
            if "token" in self.auth:
                auth_vars.extend(self._extract_template_variables(str(self.auth["token"])))
            if "value" in self.auth:
                auth_vars.extend(self._extract_template_variables(str(self.auth["value"])))
            for var in auth_vars:
                # Use custom description if provided, otherwise use default
                variables[var] = self.parameter_descriptions.get(var, "Authentication parameter")
        
        return variables
    
    def to_schema(self) -> dict:
        """Convert the ApiTool to OpenAI tool schema format using template variables."""
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
        
        # Get all template variables
        template_vars = self._get_all_template_variables()
        
        # Add each variable as a parameter
        for var_name, var_description in template_vars.items():
            schema["function"]["parameters"]["properties"][var_name] = {
                "type": "string",
                "description": var_description
            }
            # All template variables are required
            schema["function"]["parameters"]["required"].append(var_name)
        
        return schema
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Make an HTTP API request with the provided parameters.
        
        Args:
            **kwargs: Dynamic parameters that can be used to populate URL templates,
                     body templates, query parameters, and headers.
        
        Returns:
            Dict[str, Any]: API response data including status, headers, and content
        """
        try:
            import requests
            
            # 1. Build the URL with template substitution
            final_url = self.url.format(**kwargs)
            
            # 2. Build query parameters
            params = {}
            if self.query_params_template:
                for key, value_template in self.query_params_template.items():
                    if isinstance(value_template, str) and '{' in value_template:
                        params[key] = value_template.format(**kwargs)
                    else:
                        params[key] = value_template
            
            # 3. Build headers with any template substitution
            headers = self.headers.copy()
            for key, value in headers.items():
                if isinstance(value, str) and '{' in value:
                    headers[key] = value.format(**kwargs)
            
            # 4. Handle authentication
            if self.auth:
                auth_type = self.auth.get("type", "").lower()
                if auth_type == "bearer" and "token" in self.auth:
                    token_template = self.auth["token"]
                    if isinstance(token_template, str) and '{' in token_template:
                        token = token_template.format(**kwargs)
                    else:
                        token = token_template
                    headers["Authorization"] = f"Bearer {token}"
                elif auth_type == "api_key" and "key" in self.auth and "value" in self.auth:
                    key_name = self.auth["key"]
                    value_template = self.auth["value"]
                    if isinstance(value_template, str) and '{' in value_template:
                        value = value_template.format(**kwargs)
                    else:
                        value = value_template
                    headers[key_name] = value
            
            # 5. Build request body - use safer template substitution
            data = None
            if self.body_template and self.method in ["POST", "PUT", "PATCH"]:
                # Use a safer approach for JSON templates
                body_str = self.body_template
                for key, value in kwargs.items():
                    placeholder = "{" + key + "}"
                    body_str = body_str.replace(placeholder, str(value))
                
                try:
                    data = json.loads(body_str)
                except json.JSONDecodeError:
                    data = body_str
            
            # 6. Make the HTTP request
            response = requests.request(
                method=self.method,
                url=final_url,
                headers=headers,
                params=params,
                json=data if isinstance(data, dict) else None,
                data=data if isinstance(data, str) else None,
                timeout=self.timeout
            )
            
            # 7. Process the response
            result = {
                "status_code": response.status_code,
                "url": response.url,
                "headers": dict(response.headers)
            }
            
            # Try to parse JSON response, fall back to text
            try:
                result["data"] = response.json()
            except json.JSONDecodeError:
                result["data"] = response.text
            
            # 8. Check for HTTP errors and include in result
            if not response.ok:
                result["error"] = f"HTTP {response.status_code}: {response.reason}"
            
            return result
            
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}


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
            data={"content": full_response}
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
        ask_user = FunctionTool(
            name="ask_user",
            description="Ask the user for additional information when you need clarification or missing parameters",
            function=lambda question: {"question": question, "needs_user_input": True}
        )
        tools.append(ask_user)  # Add ask_user tool to the tools list
        super().__init__(name, description, provider, tools)
        self._token_buffer = []  # Buffer for collecting tokens when streaming

    def _planning_phase(self, messages: Messages):
        """Planning phase to analyze the question and plan the next steps
        Args:
            messages (Messages): The current message context
            
        Returns:
            str: The planning response
        """
        yield AgentEvent(
            AgentEventType.PROGRESS,
            message="Planning...",
            data={}
        )        
        session_context = f"*Session context*\nCurrent date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        system_message = f"""{session_context}\n
You are a task analysis and planning agent for a ReAct system.

Instructions:
- Analyze the user's request and determine the best approach to fulfill it
- If the request is clear and you have sufficient information, plan direct tool usage
- Only ask for clarification when truly essential information is missing AND cannot be reasonably inferred
- Consider if reasonable defaults or assumptions can be made instead of asking

Planning Process:
1. Rewrite the user's input with context from previous conversation turns
2. Assess completeness: Can this be executed with available information?
3. If YES: Plan the tool sequence needed to complete the task
4. If NO: Identify ONLY the critical missing information that blocks execution
5. Choose tools and specify their parameters
6. For web search tools, use them when current/recent information is needed

Guidelines:
- Prefer action over questions when reasonable assumptions can be made
- Simple, common requests usually don't need clarification
- Focus on what CAN be done rather than what's missing
- Only use ask_user tool when absolutely necessary

Output your analysis and execution plan (Not the response to the user):"""
        
        messages.add_system_message(system_message)
                
        result = self.provider.generate(
            messages.get_messages(),
            tools=self.tool_schemas,
            tool_choice="none",
        )

        content = self.provider.extract_content(result)
        #messages.clear()
        messages.add_system_message(system_message)
        messages.add_assistant_message(content)
        yield AgentEvent(
            AgentEventType.THINKING,
            message="Planning complete.",
            data={"content": content}
        )

    def _setup_action_phase(self, messages: Messages):
        """Setup action phase to prepare the messages for the action phase.
        
        Args:
            messages (Messages): The current message context
        """
        messages.add_system_message(f"""Based on your planning response above, you have identified the tools needed to solve the problem.                                            
            Now take appropriate action to solve the problem. You can use multiple tools in sequence if needed.
            If you need to use multiple tools, execute them one at a time and use the results from previous tools to inform subsequent tool calls.            
            Take only one action at a time, wait for the results, then decide on the next action based on those results.""")
        
    def _take_action_phase(self, messages: Messages):
        """Action phase to execute the tools based on the planning response.
        
        Args:
            messages (Messages): The current message context
            
        Returns:
            str: The action response
            List[any]: The tool calls
        """

        result = self.provider.generate(
            messages.get_messages(),
            tools=self.tool_schemas,
            tool_choice="auto",            
        )
        
        content = self.provider.extract_content(result)
        tool_calls = self.provider.extract_tool_calls(result)        
       
        messages.add_message(Message(role="assistant", content=content, tool_calls=tool_calls))
        return content, tool_calls

    def _observe_phase(self, messages: Messages):
        """Observation phase to analyze the results of the tool calls.
        
        Args:
            messages (Messages): The current message context
            
        Returns:
            str: The observation response
        """
        messages.add_system_message( """Based on the tool results above, decide if:
                1) You need to make additional tool calls to complete the task (if so, make the next appropriate tool call), or
                2) You have all the information needed to provide a final answer (if so, provide your comprehensive final response with no tool calls).
                
                Make sure to review all previous tool results when making your decision.
                Make sure to that the final answer is conversational and easy to understand.
                If any tools threw an error, attempt to fix the error and re-run the tool, unless the error was due to hitting the rate limit or similar issue.""")
   
    def _get_tool_parts(self, tool_call):
        """Extract the function name and arguments from a tool call.
        
        Args:
            tool_call (any): The tool call object
        
        Returns:
            tuple: A tuple containing the function name and arguments
        """
        if hasattr(tool_call, "function"):
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments) if type(tool_call.function.arguments)== str else tool_call.function.arguments
            tool_call_id = tool_call.id
        else:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"]) if type(tool_call["function"]["arguments"])== str else tool_call["function"]["arguments"]
            tool_call_id = tool_call.get("id", "0")
        
        return function_name, function_args, tool_call_id
        
    def _execute_tools(self, messages: Messages, tool_calls: any, iteration: int) -> Generator[AgentEvent, None, None]:
        """Execute the tools based on the tool calls.
        
        Args:
            messages (Messages): The current message context
            tool_calls (any): The tool calls to execute
        """
        # Process each tool call in current iteration        
        for tool_idx, tool_call in enumerate(tool_calls):
            try:
                
                function_name, function_args, tool_call_id = self._get_tool_parts(tool_call)
                
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
                                
                if function_name in self.tool_map:
                    result = self.tool_map.get(function_name).run(**function_args)                    
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
                    f"{tool_call_id}",
                    function_name,
                    json.dumps(result)
                )
                messages.add_message(Message(**tool_response))
                
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
                messages.add_message(Message(**tool_response))

    def _synthesize_final_answer(self, messages: Messages, **kwargs) -> Generator[AgentEvent, None, None]:
        """Synthesize the final answer based on the messages.
        
        Args:
            messages (Messages): The current message context
            
        Yields:
            AgentEvent: Events containing the final synthesized answer
        """
        stream = kwargs.pop("stream", False)
        yield AgentEvent(
            AgentEventType.PROGRESS,
            message="Writintg final answer",
        )
        messages.add_system_message(f"This is the final answer phase. Synthesize the final answer from the messages above to answer the question. This should be formatted as the response to user. Use markdown formatting where appropriate.")
        
        final_output = ""
        if stream:
            # If streaming is enabled, use the streaming response method            
            for chunk in self.provider.generate(
                messages.get_messages(),
                stream=True,
                **kwargs
            ):
                if hasattr(chunk, "choices") and chunk.choices:
                    # Extract delta content if available
                    if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                        token = chunk.choices[0].delta.content
                        if token:
                            final_output += token
                            yield AgentEvent(
                                AgentEventType.PARTIAL_RESPONSE,
                                data={"token": token}
                            )
                    # Some providers might provide different formats
                    elif hasattr(chunk.choices[0], "text"):
                        token = chunk.choices[0].text
                        if token:
                            final_output += token
                            yield AgentEvent(
                                AgentEventType.PARTIAL_RESPONSE,
                                data={"token": token}
                            )
        else:    
            response = self.provider.generate(
                messages.get_messages()            
            )
            
            final_output = self.provider.extract_content(response)
        
        yield AgentEvent(
            AgentEventType.FINAL_RESPONSE,
            message="Task complete",
            data={"content": final_output}
        )

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
        
        messages = messages.copy()

        # Execute the planning phase
        yield from self._planning_phase(messages)
        self._setup_action_phase(messages)

        iterations = 0
        max_iterations = 5
        while iterations < max_iterations:    
                yield AgentEvent(
                    AgentEventType.PROGRESS,
                    message="Action phase...",
                    data={}
                )

                content, tool_calls = self._take_action_phase(messages)               

                yield AgentEvent(
                    AgentEventType.THINKING,
                    message="Action phase complete",
                    data={"content": content, "tool_calls": f"{tool_calls}"}
                )

                if tool_calls:
                    # get tool call parts
                    function_name, function_args, tool_call_id = self._get_tool_parts(tool_calls[0])
                    if function_name == "ask_user":
                        # If the first tool call is to ask the user, we need to handle that
                        yield AgentEvent(
                            AgentEventType.PROGRESS,
                            message="Asking user for additional information",
                            data={"content": content}
                        )
                        messages.pop()  # Remove the last assistant message                        
                        yield from self._synthesize_final_answer(messages, stream=stream)
                        return                                                
                    yield from self._execute_tools(messages, tool_calls, iterations)                    
                    self._observe_phase(messages)                    
                else:
                    yield AgentEvent(
                        AgentEventType.THINKING,
                        message="No tools needed, skipping observation phase",
                        data={"content": content}
                    )
                    yield from self._synthesize_final_answer(messages, stream=stream)
                    return

                if iterations == max_iterations:
                    yield AgentEvent(
                        AgentEventType.PROGRESS,
                        message="Max iterations reached. Stopping execution.",
                        data={"content": "Max iterations reached."}
                    )
                    yield from self._synthesize_final_answer(messages, stream=stream)
                    return
                
                iterations += 1


# class KnowledgeAgent(Agent):
#     """Agent that uses a web_search tool to gather additional information from the web before answering the question."""
#     def __init__(self, name: str, description: str, provider: LLMProvider, tools: List[Tool]=[]):
#         from .tools import web_search
#         tools.append(web_search)
#         super().__init__(name, description, provider, tools)

#     def _parse_search_queries(self, content: str) -> List[str]:
#         """Parse the search queries from the content.
        
#         Args:
#             content (str): The content containing search queries in JSON format
            
#         Returns:
#             List[str]: A list of search queries extracted from the content
#         """
#         try:
#             # Attempt to parse the content as JSON
#             queries = json.loads(content)
#             if isinstance(queries["queries"], list):
#                 return queries["queries"]
#             else:
#                 return []
#         except json.JSONDecodeError:
#             return []

#     def _generate_search_queries(self, messages: Messages) -> List[str]:
#         """Generate enhanced search queries based on the conversation messages.
        
#         Args:
#             messages (Messages): The current message context
            
#         Returns:
#             List[str]: A list of search queries generated from the conversation messages            
#         """
     
#         session_context = f"*Session context*\nCurrent date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
#         system_message = f"""{session_context}\n
# You are a search query generation agent for a Knowledge Agent system.

# Instructions:
# - Analyze the user's request and determine the best approach to fulfill it

# Process:
# 1. Rewrite the user's input with context from previous conversation turns
# 2. Assess completeness: Can this be executed with available information?
# 3. Generate a list of 5 search queries to find relevant information.

# Output:
# - Analysis and a list of search queries that can be used to gather additional information from the web. Response MUST use JSON format. Example:
# {{
#     "analysis": "The user is asking about the latest news on AI advancements.",
#     "queries": [
#         "latest AI advancements",
#         "AI news October 2023",
#         "AI breakthroughs 2023",
#         "top AI technologies 2023",
#         "AI industry trends"
#     ]
# }}
# """
#         messages = messages.copy()
#         messages.add_system_message(system_message)
                
#         result = self.provider.generate(
#             messages.get_messages(),
#             tools=self.tool_schemas,
#             tool_choice="none",
#         )

#         content = self.provider.extract_content(result)
#         return content, self._parse_search_queries(content)



#     def run(self, **kwargs) -> Generator[AgentEvent, None, None]:
#         """Run the Knowledge Agent with the given input.
        
#         Args:
#             **kwargs: Additional arguments including:
#                 - messages: List of conversation messages
#                 - stream: Boolean indicating whether to stream the final response
                
#         Yields:
#             AgentEvent: Structured event objects providing updates on agent progress
#         """
#         stream = kwargs.pop("stream", False)
#         messages = kwargs.pop("messages", None)
        
#         if messages is None or not isinstance(messages, Messages):
#             yield AgentEvent(AgentEventType.ERROR, message="Invalid messages provided.")
#             return           

#         yield AgentEvent(
#             AgentEventType.PROGRESS,
#             message="Generating search queries",
#             data={}
#         )
#         content, queries = self._generate_search_queries(messages) 
#         yield AgentEvent(
#             AgentEventType.PROGRESS,
#             message="Generated search queries",
#             data={"content": content}
#         )
#         yield AgentEvent(
#             AgentEventType.PROGRESS,
#             message="Generated search queries",
#             data={"queries": queries}
#         )

#         if not queries:
#             yield AgentEvent(
#                 AgentEventType.FINAL_RESPONSE,
#                 message="No search queries generated. Unable to proceed.",
#                 data={"content": "No search queries generated. Unable to proceed."}
#             )
#             return
        
#         # execute the web search tool for each query
#         for query in queries:
#             yield AgentEvent(
#                 AgentEventType.TOOL_CALL,
#                 message=f"Executing web search for query: {query}",
#                 data={"query": query}
#             )
            
#             web_search_tool = self.tool_map.get("web_search")
#             if not web_search_tool:
#                 yield AgentEvent(
#                     AgentEventType.ERROR,
#                     message="Web search tool not found.",
#                     data={}
#                 )
#                 return
            
#             result = web_search_tool.run(q=query, count=10)
            
#             yield AgentEvent(
#                 AgentEventType.TOOL_RESULT,
#                 message=f"Web search completed for query: {query}",
#                 data={"result": result}
#             )
            
#             # Add the result to the messages
#             messages.add_message(Message(role="assistant", content=json.dumps(result)))
        
        