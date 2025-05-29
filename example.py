"""
Basic example showing how to use the CPYO framework with a simple tool.
"""
from cpyo import OpenAIProvider, FunctionTool, Agent, ReActAgent, Messages, PythonTool, AgentEventType, tool, ApiTool
from cpyo.tools import traffic_checker, web_search, python_executor, get_joke
from dotenv import load_dotenv
from colorama import Fore, Back, Style, init

# Initialize colorama for colored output
init()
load_dotenv()


## Tasks API api tools section

# Task Manager API Tools
task_api_base = "http://localhost:8000"

# Create a new task
create_task = ApiTool(
    name="create_task",
    description="Create a new task and assign it to person. assigned_to must be person's name (Ask users name if you do not know.).",
    url=f"{task_api_base}/tasks/",
    method="POST",
    headers={"Content-Type": "application/json"},
    body_template="""{
        "title": "{title}",
        "description": "{description}",
        "assigned_to": "{assigned_to}"
    }"""
)
print(create_task.to_schema())
# Get all tasks with optional filters
get_tasks = ApiTool(
    name="get_tasks",
    description="Get all tasks, optionally filtered by person or completion status",
    url=f"{task_api_base}/tasks/",
    method="GET",
    query_params_template={
        "assigned_to": "{assigned_to}",
        "completed": "{completed}"
    }
)

# Get a specific task by ID
get_task = ApiTool(
    name="get_task",
    description="Get details of a specific task by its ID",
    url=f"{task_api_base}/tasks/{{task_id}}",
    method="GET"
)

# Update a task
update_task = ApiTool(
    name="update_task",
    description="Update task details like title, description, or assignee",
    url=f"{task_api_base}/tasks/{{task_id}}",
    method="PUT",
    headers={"Content-Type": "application/json"},
    body_template="""{
        "title": "{title}",
        "description": "{description}",
        "assigned_to": "{assigned_to}"
    }"""
)

# Mark task as completed
complete_task = ApiTool(
    name="complete_task",
    description="Mark a task as completed",
    url=f"{task_api_base}/tasks/{{task_id}}/complete",
    method="PATCH"
)

# Reopen a completed task
reopen_task = ApiTool(
    name="reopen_task",
    description="Reopen a previously completed task",
    url=f"{task_api_base}/tasks/{{task_id}}/reopen",
    method="PATCH"
)

# Delete a task
delete_task = ApiTool(
    name="delete_task",
    description="Delete a task permanently",
    url=f"{task_api_base}/tasks/{{task_id}}",
    method="DELETE"
)

# Get task statistics
get_task_stats = ApiTool(
    name="get_task_stats",
    description="Get overall task statistics and per-person breakdown",
    url=f"{task_api_base}/stats/",
    method="GET"
)


##


def main():
    messages = Messages()    
    messages.add_system_message("You are a helpful assistant.")
    
    provider = OpenAIProvider()
    agent = ReActAgent(
        name="ReActAgent",
        description="Agent that can check traffic data, execute python code, search the web, and make API calls.",
        provider=provider,
        tools=[traffic_checker, web_search, python_executor, get_joke, 
               create_task, get_tasks, get_task, update_task, complete_task, 
               reopen_task, delete_task, get_task_stats]        
    )
    
    while True:
        # Get user input
        user_input = input(Back.BLUE + "You: " + Style.RESET_ALL)
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        # Add user message to conversation
        messages.add_user_message(user_input)
        
        # Run the agent
        try:
            # Run the agent
            for event in agent.run(messages=messages, stream=True, model="gpt-4.1-mini", temperature=0.7):
                if event.event_type == AgentEventType.THINKING:
                    print(Fore.LIGHTBLACK_EX + f"🧠 Thinking: {event.message}, {event.data}" + Fore.RESET)
                
                elif event.event_type == AgentEventType.TOOL_CALL:
                    print(Fore.LIGHTBLACK_EX + f"🔧 Tool: {event.data}" + Fore.RESET)
                    
                elif event.event_type == AgentEventType.TOOL_RESULT:
                    result = f"{event.data['result']}"                    
                    print(Fore.LIGHTBLACK_EX + f"📋 Result: {event.data['tool_name']}, {result[:200]}..." + Fore.RESET)
                    
                elif event.event_type == AgentEventType.PARTIAL_RESPONSE:
                    # Keep this visible as requested
                    token = event.data["token"]
                    print(Fore.CYAN + token + Fore.RESET, end="", flush=True)
                    
                elif event.event_type == AgentEventType.FINAL_RESPONSE:
                    print()
                    #print(event.data["response"])
                    messages.add_assistant_message(event.data["content"])
                    
                elif event.event_type == AgentEventType.ERROR:
                    print(Fore.LIGHTRED_EX + f"⚠️ Error: {event.message} {event.data}" + Fore.RESET)
                    
                elif event.event_type == AgentEventType.PROGRESS:
                    print(Fore.LIGHTBLACK_EX + f"⏱️  Progress: {event.message} {event.data}" + Fore.RESET)
            
        except Exception as e:
            print(Fore.LIGHTRED_EX + f"⚠️ Error: {str(e)}" + Fore.RESET)

if __name__ == "__main__":
    main()