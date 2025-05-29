# CPYO Agentic Framework

<img src="images/logo.png" alt="CPYO Logo" width="200"/>

## Overview

CPYO is a personal learning framework for building and experimenting with agentic AI systems. This project is in **active development** and serves as a hands-on tool for understanding the architecture and capabilities of agent-based AI systems.

> **Note**: This is a learning project and not intended for production use. Features and APIs may change frequently as development progresses.

## Purpose

This framework is being developed to:

- Learn about agentic AI architecture and design patterns
- Experiment with different agent types and capabilities
- Provide a playground for testing various LLM integrations
- Explore tool-calling mechanisms and agent reasoning

## Features

- 🤖 Multiple agent types (including ReAct agents)
- 🔧 Tool integration framework (Function tools, Python execution)
- 🔄 Event-driven architecture for agent interactions
- 🧠 Memory management for conversation history
- 🔌 Flexible provider system (currently supporting OpenAI)
- 🌐 Built-in tools for web search, calculations, and more

## Quick Start

```python
from cpyo import OpenAIProvider, ReActAgent, Messages, AgentEventType, tool
from cpyo.tools import python_executor, web_search, traffic_checker
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create a simple tool using the @tool decorator
@tool
def calculator(operation: str, a: float, b: float) -> dict:
    """Perform basic mathematical operations.

    Parameters:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number

    Returns:
        Dictionary containing the result of the operation
    """
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }
    return {"result": operations.get(operation, "Invalid operation")}

# Initialize provider (requires OPENAI_API_KEY environment variable)
provider = OpenAIProvider()

# Create an agent with multiple tools
agent = ReActAgent(
    name="MathHelper",
    description="An assistant that can perform calculations, execute Python code, and search the web",
    provider=provider,
    tools=[calculator, python_executor, web_search]
)

# Set up conversation
messages = Messages()
messages.add_system_message("You are a helpful assistant that can perform calculations and answer questions.")

# Interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break

    messages.add_user_message(user_input)

    # Run the agent and handle events
    for event in agent.run(messages=messages, stream=True, model="gpt-4o-mini"):
        if event.event_type == AgentEventType.THINKING:
            print(f"🧠 Thinking: {event.message}")

        elif event.event_type == AgentEventType.TOOL_CALL:
            print(f"🔧 Using tool: {event.data}")

        elif event.event_type == AgentEventType.PARTIAL_RESPONSE:
            print(event.data["token"], end="", flush=True)

        elif event.event_type == AgentEventType.FINAL_RESPONSE:
            print()  # New line after streaming response
            messages.add_assistant_message(event.data["content"])

        elif event.event_type == AgentEventType.ERROR:
            print(f"⚠️ Error: {event.message}")
```

### Environment Setup

Before running, make sure to set up your environment variables:

```bash
# Required for OpenAI integration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For web search functionality
BRAVE_API_KEY=your_brave_search_api_key_here

# Optional: For traffic checking functionality
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

### Creating Custom Tools

You can easily create custom tools using the `@tool` decorator:

```python
@tool
def weather_check(location: str) -> dict:
    """Check weather for a given location.

    Parameters:
        location: The location to check weather for

    Returns:
        Dictionary containing weather information
    """
    # Your implementation here
    return {"location": location, "weather": "sunny"}
```

## Project Status

- [x] Basic agent framework
- [x] Tool calling integration
- [x] Event system for monitoring agent activity
- [x] OpenAI integration
- [ ] More agent types (in progress)
- [ ] Additional LLM providers
- [ ] Advanced memory management
- [ ] Agent benchmarking tools

## Development

This project is being actively developed. Feel free to explore the code and experiment with the framework. Documentation will improve as the project matures.

## TODO

- Implement agent type tool calling
- Add more comprehensive documentation
- Create additional examples
- Add tests

## ApiTool Documentation

The `ApiTool` class provides a powerful way to integrate HTTP API calls into your agent's toolkit. It supports template-based parameter substitution, various authentication methods, and flexible request configuration.

### Basic Usage

```python
from cpyo import ApiTool

# Simple GET request
weather_api = ApiTool(
    name="get_weather",
    description="Get current weather for a city",
    url="https://api.openweathermap.org/data/2.5/weather",
    method="GET",
    query_params_template={
        "q": "{city}",
        "appid": "{api_key}",
        "units": "metric"
    }
)

# Usage in agent
agent = ReActAgent(
    name="WeatherAgent",
    description="Agent that can check weather",
    provider=provider,
    tools=[weather_api]
)
```

### Template Variables

ApiTool uses `{variable_name}` syntax for template substitution. Variables can be used in:

- URL paths
- Query parameters
- Request body
- Headers
- Authentication values

### Authentication Examples

#### Bearer Token Authentication

```python
# API with Bearer token
api_with_bearer = ApiTool(
    name="github_user",
    description="Get GitHub user information",
    url="https://api.github.com/users/{username}",
    method="GET",
    auth={
        "type": "bearer",
        "token": "{github_token}"
    }
)
```

#### API Key Authentication

```python
# API key in header
api_with_key = ApiTool(
    name="news_api",
    description="Get latest news",
    url="https://newsapi.org/v2/top-headlines",
    method="GET",
    auth={
        "type": "api_key",
        "key": "X-API-Key",
        "value": "{news_api_key}"
    },
    query_params_template={
        "country": "{country}",
        "category": "{category}"
    }
)
```

### POST Request with JSON Body

```python
# Creating a new resource
create_user_api = ApiTool(
    name="create_user",
    description="Create a new user account",
    url="https://api.example.com/users",
    method="POST",
    headers={
        "Content-Type": "application/json"
    },
    body_template='''{
        "username": "{username}",
        "email": "{email}",
        "firstName": "{first_name}",
        "lastName": "{last_name}"
    }''',
    auth={
        "type": "bearer",
        "token": "{admin_token}"
    }
)
```

### RESTful CRUD Operations

```python
# Read operation
get_user_api = ApiTool(
    name="get_user",
    description="Get user by ID",
    url="https://api.example.com/users/{user_id}",
    method="GET",
    auth={"type": "bearer", "token": "{api_token}"}
)

# Update operation
update_user_api = ApiTool(
    name="update_user",
    description="Update user information",
    url="https://api.example.com/users/{user_id}",
    method="PUT",
    headers={"Content-Type": "application/json"},
    body_template='''{
        "email": "{email}",
        "firstName": "{first_name}",
        "lastName": "{last_name}"
    }''',
    auth={"type": "bearer", "token": "{api_token}"}
)

# Delete operation
delete_user_api = ApiTool(
    name="delete_user",
    description="Delete user by ID",
    url="https://api.example.com/users/{user_id}",
    method="DELETE",
    auth={"type": "bearer", "token": "{api_token}"}
)
```

### Search and Filter APIs

```python
# Complex search with multiple parameters
search_products_api = ApiTool(
    name="search_products",
    description="Search products with filters",
    url="https://api.store.com/products/search",
    method="GET",
    query_params_template={
        "q": "{search_term}",
        "category": "{category}",
        "min_price": "{min_price}",
        "max_price": "{max_price}",
        "sort": "{sort_order}",
        "limit": "{limit}"
    },
    headers={
        "X-Store-ID": "{store_id}"
    }
)
```

### External Service Integration Examples

#### Slack Integration

```python
# Send Slack message
slack_message_api = ApiTool(
    name="send_slack_message",
    description="Send message to Slack channel",
    url="https://slack.com/api/chat.postMessage",
    method="POST",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer {slack_token}"
    },
    body_template='''{
        "channel": "{channel}",
        "text": "{message}",
        "username": "{bot_name}"
    }'''
)
```

#### Email Service

```python
# Send email via API
send_email_api = ApiTool(
    name="send_email",
    description="Send email using email service API",
    url="https://api.emailservice.com/v1/send",
    method="POST",
    headers={
        "Content-Type": "application/json"
    },
    auth={
        "type": "api_key",
        "key": "X-API-Key",
        "value": "{email_api_key}"
    },
    body_template='''{
        "to": "{recipient}",
        "subject": "{subject}",
        "html": "{html_content}",
        "from": "{sender_email}"
    }'''
)
```

#### Database API

```python
# Query database via REST API
query_database_api = ApiTool(
    name="query_users",
    description="Query users from database",
    url="https://db-api.example.com/query",
    method="POST",
    headers={"Content-Type": "application/json"},
    auth={"type": "bearer", "token": "{db_token}"},
    body_template='''{
        "table": "users",
        "where": {
            "status": "{status}",
            "created_after": "{date}"
        },
        "limit": {limit}
    }'''
)
```

### Error Handling

ApiTool automatically handles common HTTP errors and returns structured responses:

```python
# The tool returns a dictionary with:
{
    "status_code": 200,
    "url": "https://api.example.com/endpoint",
    "headers": {"content-type": "application/json"},
    "data": {...},  # Parsed JSON or raw text
    "error": "HTTP 404: Not Found"  # Only present for HTTP errors
}
```

### Advanced Configuration

```python
# API with custom timeout and headers
advanced_api = ApiTool(
    name="advanced_api_call",
    description="API call with advanced configuration",
    url="https://api.example.com/data/{endpoint}",
    method="POST",
    timeout=60,  # Custom timeout in seconds
    headers={
        "User-Agent": "MyAgent/1.0",
        "Accept": "application/json",
        "X-Custom-Header": "{custom_value}"
    },
    query_params_template={
        "version": "v2",
        "format": "json",
        "filter": "{filter_criteria}"
    },
    body_template='''{
        "query": "{search_query}",
        "options": {
            "include_metadata": true,
            "max_results": {max_results}
        }
    }''',
    auth={
        "type": "bearer",
        "token": "{access_token}"
    }
)
```

### Using ApiTool in Agents

```python
from cpyo import OpenAIProvider, ReActAgent, Messages

# Create multiple API tools
tools = [
    weather_api,
    news_api,
    slack_message_api,
    search_products_api
]

# Create agent with API tools
api_agent = ReActAgent(
    name="APIAgent",
    description="Agent that can interact with various APIs",
    provider=OpenAIProvider(),
    tools=tools
)

# Use the agent
messages = Messages()
messages.add_user_message("Check the weather in London and send a summary to #general channel on Slack")

for event in api_agent.run(messages=messages):
    # Handle events...
    pass
```

### Best Practices

1. **Use descriptive names and descriptions** for your API tools to help the LLM understand when to use them.

2. **Template variables are automatically detected** and become required parameters in the tool schema.

3. **Handle sensitive data carefully** - use environment variables for API keys and tokens.

4. **Test your templates** before deploying to ensure proper JSON formatting.

5. **Set appropriate timeouts** for external API calls to prevent hanging requests.

6. **Use specific parameter descriptions** to guide the LLM in providing correct values.

### Environment Variables Example

```bash
# API Keys and tokens
WEATHER_API_KEY=your_weather_api_key
GITHUB_TOKEN=your_github_token
NEWS_API_KEY=your_news_api_key
SLACK_TOKEN=your_slack_bot_token
EMAIL_API_KEY=your_email_service_key
```

The ApiTool class makes it easy to integrate any REST API into your agent's capabilities, enabling powerful workflows that combine LLM reasoning with real-world data and actions.
