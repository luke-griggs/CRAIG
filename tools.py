#!/usr/bin/env python3
"""
Tool functions for the voice assistant.
Implements various tools that can be called by the LLM.
"""

import os
import json
from typing import Dict, Any

import requests
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def rainbow_text_tool(message: str = "Rainbow text activated!") -> str:
    """
    Tool that outputs colorful rainbow text to the console.
    
    Args:
        message: The message to display in rainbow colors
        
    Returns:
        Confirmation message that the tool was called
    """
    # Rainbow colors using colorama
    colors = [
        Fore.RED,
        Fore.YELLOW, 
        Fore.GREEN,
        Fore.CYAN,
        Fore.BLUE,
        Fore.MAGENTA
    ]
    
    print(f"\n{'='*50}")
    print(f"{Fore.WHITE}🌈 RAINBOW TEXT TOOL ACTIVATED 🌈{Style.RESET_ALL}")
    print(f"{'='*50}")
    
    # Print each character in a different color
    rainbow_message = ""
    for i, char in enumerate(message):
        color = colors[i % len(colors)]
        rainbow_message += f"{color}{char}{Style.RESET_ALL}"
    
    print(rainbow_message)
    print(f"{'='*50}\n")
    
    return f"Rainbow text tool executed successfully! Displayed: '{message}'"


def weather_lookup_tool(location: str, units: str = "metric") -> str:
    """Fetch current weather details for the requested location."""
    if not location:
        return "Weather tool error: a location is required."

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather tool error: OPENWEATHER_API_KEY is not set in the environment."

    units = units.lower()
    if units not in {"metric", "imperial", "standard"}:
        return "Weather tool error: units must be 'metric', 'imperial', or 'standard'."

    unit_label = {
        "metric": "C",
        "imperial": "F",
        "standard": "K"
    }[units]

    params = {
        "q": location,
        "appid": api_key,
        "units": units
    }

    try:
        response = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return f"Weather tool error: API request failed (status {status})."
    except requests.exceptions.RequestException as exc:
        return f"Weather tool error: network problem ({exc})."
    except ValueError:
        return "Weather tool error: invalid response received from the API."

    weather = payload.get("weather", [{}])[0].get("description", "unknown conditions")
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    name = payload.get("name", location)

    temperature = main.get("temp")
    feels_like = main.get("feels_like")
    humidity = main.get("humidity")
    wind_speed = wind.get("speed")

    summary_parts = [f"Current weather in {name}: {weather}"]

    if temperature is not None:
        summary_parts.append(f"temperature {temperature:.1f} {unit_label}")
    if feels_like is not None:
        summary_parts.append(f"feels like {feels_like:.1f} {unit_label}")
    if humidity is not None:
        summary_parts.append(f"humidity {humidity}%")
    if wind_speed is not None:
        speed_unit = "m/s" if units != "imperial" else "mph"
        summary_parts.append(f"wind {wind_speed:.1f} {speed_unit}")

    report = ", ".join(summary_parts)

    print(f"\n{'='*50}")
    print(f"{Fore.WHITE}☁️ WEATHER REPORT ☁️{Style.RESET_ALL}")
    print(f"{'='*50}")
    print(f"{Fore.CYAN}{report}{Style.RESET_ALL}")
    print(f"{'='*50}\n")

    return report

# Tool definitions for Groq API
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rainbow_text_tool",
            "description": "Display colorful rainbow text in the console. Use this when the user asks for something colorful, fun, or when you want to add visual flair to responses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to display in rainbow colors. If not specified, uses default message."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather_lookup_tool",
            "description": "Get the current weather for a city using OpenWeatherMap. Provide city name and optional units (metric, imperial, standard).",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, optionally with country code (e.g., 'Paris,FR')."
                    },
                    "units": {
                        "type": "string",
                        "description": "Measurement units: metric (C), imperial (F), or standard (K). Defaults to metric."
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Map of tool names to functions
TOOL_FUNCTIONS = {
    "rainbow_text_tool": rainbow_text_tool,
    "weather_lookup_tool": weather_lookup_tool
}

def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool call by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments to pass to the tool
        
    Returns:
        Result from the tool execution
    """
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{tool_name}'"
    
    try:
        tool_function = TOOL_FUNCTIONS[tool_name]
        result = tool_function(**arguments)
        return result
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

if __name__ == "__main__":
    # Test the rainbow text tool
    print("Testing rainbow text tool...")
    result = rainbow_text_tool("Hello, colorful world! 🌈✨")
    print(f"Result: {result}")
