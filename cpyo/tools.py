import datetime
import os
from typing import Any, Dict

import googlemaps
import requests
from .core import PythonTool, tool


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for real-time information on a query using Brave Search.
    
    Parameters:
        query: Search query string. Has to be a well-formed question or search term.

    Returns:
        A dictionary containing the search results, including titles, descriptions, URLs, and published dates.
        If an error occurs, it returns a dictionary with an error message.
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


@tool
def traffic_checker(current_location: str, destination: str):
    """Get traffic data from the current location to the destination using Distance Matrix API. Users home country is Ireland.
    
    Parameters:
        current_location: Current location of the user
        destination: Destination location
    """
    # Initialize Google Maps client
    if not os.getenv("GOOGLE_MAPS_API_KEY"):
        return {"error": "Please set the GOOGLE_MAPS_API_KEY environment variable."}
    gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))    
    # Get traffic data
    try:
        now = datetime.datetime.now()
        directions_result = gmaps.distance_matrix(current_location, destination, mode="driving", departure_time=now)
        return directions_result
    except Exception as e:
        return {"error": f"Error fetching traffic data: {str(e)}"}


python_executor = PythonTool(
    name="python_executor",
    description="Execute Python code",    
)