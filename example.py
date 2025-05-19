"""
Basic example showing how to use the CPYO framework with a simple tool.
"""

import os
import json
from typing import Any, Dict

import requests
from cpyo import OpenAIProvider, FunctionTool, Agent, ReActAgent, Messages
from dotenv import load_dotenv
import time
from colorama import Fore, Back, Style, init

from cpyo.event import AgentEventType
# Initialize colorama (needed for Windows)
init()


load_dotenv()
import googlemaps
import datetime

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Please set the OPENAI_API_KEY environment variable.")
    exit(1)

def web_search_impl(query: str) -> Dict[str, Any]:
    """Search the web for real-time information on a query using Brave Search.
    
    Parameters:
        query: Search query string
    """
    try:
        # You would need to get an API key from https://brave.com/search/api/
        BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
        
        if not BRAVE_API_KEY:
            return {"error": "BRAVE_API_KEY not set in environment variables."}
        
        # Real implementation with Brave Search API
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        
        params = {
            "q": query,
            "count": 10  # Number of results to return
        }
        
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params
        )
        
        if response.status_code != 200:
            return {"error": f"Search API returned status code {response.status_code}"}
        
        data = response.json()
        results = []
        sources = []
        
        for item in data.get("web", {}).get("results", []):
            result = {
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
                "published_date": item.get("published_date", "")
            }
            results.append(result)
            sources.append(item.get("url", ""))
        final_results = {
            "query": query,
            "results": results,
            "sources": sources
        }
        return final_results
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}        

def calculator_impl(operation: str, a: float, b: float):
    """Perform a basic calculation.
    
    Parameters:
        operation: Mathematical operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
    """
    if operation == "add":
        return {"result": a + b, "operation": f"{a} + {b}"}
    elif operation == "subtract":
        return {"result": a - b, "operation": f"{a} - {b}"}
    elif operation == "multiply":
        return {"result": a * b, "operation": f"{a} * {b}"}
    elif operation == "divide":
        if b == 0:
            return {"error": "Cannot divide by zero"}
        return {"result": a / b, "operation": f"{a} / {b}"}
    else:
        return {"error": f"Unknown operation: {operation}"}

def get_traffic_data_from_current_location_to_destination_impl(current_location: str, destination: str):
    """Get traffic data from the current location to the destination using Distance Matrix API. Users home country is Ireland.
    
    Parameters:
        current_location: Current location of the user
        destination: Destination location
    """
    # Initialize Google Maps client
    gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))
    
    # Get traffic data
    try:
        now = datetime.datetime.now()
        directions_result = gmaps.distance_matrix(current_location, destination, mode="driving", departure_time=now)
        return directions_result
    except Exception as e:
        return f"Error fetching traffic data: {str(e)}"

calculator = FunctionTool(
    name="calculator",
    description="Perform basic calculations (add, subtract, multiply, divide)",
    function=calculator_impl,
)

traffic_checker = FunctionTool(
    name="get_traffic_data_from_current_location_to_destination",
    description="Get traffic data from the current location to the destination using Distance Matrix API.",
    function=get_traffic_data_from_current_location_to_destination_impl,
)

web_search = FunctionTool(
    name="web_search",
    description="Search the web for real-time, up-to-date information.",
    function=web_search_impl,
)


def main():
    memory = Messages()    
    memory.add_system_message("You are a helpful assistant that can perform calculations and check traffic data.")
    # memory.add_user_message("hats traffic like from endfield, ireland to carlow")
    provider = OpenAIProvider(api_key=api_key)
    agent = ReActAgent(
        name="ReActAgent",
        description="An agent that can perform calculations and check traffic data.",
        provider=provider,
        tools=[calculator, traffic_checker, web_search]        
    )

    
    while True:
        # Get user input
        user_input = input(Back.BLUE + "You: " + Style.RESET_ALL)
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        # Add user message to conversation
        memory.add_user_message(user_input)
        
        # Run the agent
        try:
            # Run the agent
            for event in agent.run(messages=memory, stream=True, model="gpt-4.1-mini"):
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
                    memory.add_assistant_message(event.data["response"])
                    
                elif event.event_type == AgentEventType.ERROR:
                    print(Fore.LIGHTRED_EX + f"⚠️ Error: {event.message}" + Fore.RESET)
                    
                elif event.event_type == AgentEventType.PROGRESS:
                    print(Fore.LIGHTBLACK_EX + f"⏱️  Progress: {event.message}" + Fore.RESET)
            
        except Exception as e:
            print(Fore.LIGHTRED_EX + f"⚠️ Error: {str(e)}" + Fore.RESET)

if __name__ == "__main__":
    main()