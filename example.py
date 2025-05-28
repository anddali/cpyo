"""
Basic example showing how to use the CPYO framework with a simple tool.
"""
from cpyo import OpenAIProvider, FunctionTool, Agent, ReActAgent, Messages, PythonTool, AgentEventType, tool
from cpyo.tools import traffic_checker, web_search, python_executor
from dotenv import load_dotenv
from colorama import Fore, Back, Style, init

# Initialize colorama for colored output
init()
load_dotenv()



def main():
    messages = Messages()    
    messages.add_system_message("You are a helpful assistant.")
    
    provider = OpenAIProvider()
    agent = ReActAgent(
        name="ReActAgent",
        description="Agent that can check traffic data, execute python code and search the web.",
        provider=provider,
        tools=[traffic_checker, web_search, python_executor]        
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