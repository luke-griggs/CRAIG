#!/usr/bin/env python3
"""
Tool functions for the voice assistant.
Implements various tools that can be called by the LLM.
"""

import json
from typing import Dict, Any
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
    }
]

# Map of tool names to functions
TOOL_FUNCTIONS = {
    "rainbow_text_tool": rainbow_text_tool
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


