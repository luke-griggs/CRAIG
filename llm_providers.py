import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import json
import pickle

load_dotenv()

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs):
        """Generate streaming response from LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7):
        """
        Initialize OpenAI provider.
        
        Args:
            temperature: Sampling temperature (0.0 to 2.0)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI not installed. Run: pip install openai")
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.conversation_history = []
        self.history_file = "conversation_history_openai.pkl"
        self.load_conversation_history()
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate response from OpenAI.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated response text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=kwargs.get("max_tokens", 500),
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            assistant_message = response.choices[0].message.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            # Keep history limited
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # Save updated history
            self.save_conversation_history()

            return assistant_message
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """
        Generate streaming response from OpenAI.
        
        Yields:
            Response chunks as they arrive
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
                **kwargs
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Update history after streaming completes
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": full_response})

            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # Save updated history
            self.save_conversation_history()
                
        except Exception as e:
            yield f"Error: {e}"
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.save_conversation_history()

    def save_conversation_history(self):
        """Save conversation history to file."""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.conversation_history, f)
        except Exception as e:
            print(f"Warning: Could not save conversation history: {e}")

    def load_conversation_history(self):
        """Load conversation history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    self.conversation_history = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load conversation history: {e}")
            self.conversation_history = []


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.7):
        """
        Initialize Anthropic provider.
        
        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514")
            temperature: Sampling temperature (0.0 to 1.0)
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic not installed. Run: pip install anthropic")
        
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.conversation_history = []
        self.history_file = "conversation_history_anthropic.pkl"
        self.load_conversation_history()
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate response from Claude.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated response text
        """
        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 500),
                temperature=self.temperature,
                system=system_prompt or "You are a helpful voice assistant.",
                messages=messages
            )
            
            assistant_message = response.content[0].text

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            # Keep history limited
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # Save updated history
            self.save_conversation_history()

            return assistant_message
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """
        Generate streaming response from Claude.
        
        Yields:
            Response chunks as they arrive
        """
        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": prompt})
        
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 500),
                temperature=self.temperature,
                system=system_prompt or "You are a helpful voice assistant.",
                messages=messages
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
                    full_response += text
                    yield text
                
                # Update history after streaming
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": full_response})

                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]

                # Save updated history
                self.save_conversation_history()
                    
        except Exception as e:
            yield f"Error: {e}"
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.save_conversation_history()

    def save_conversation_history(self):
        """Save conversation history to file."""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.conversation_history, f)
        except Exception as e:
            print(f"Warning: Could not save conversation history: {e}")

    def load_conversation_history(self):
        """Load conversation history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    self.conversation_history = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load conversation history: {e}")
            self.conversation_history = []


class GroqProvider(LLMProvider):
    """Groq-hosted model provider."""

    def __init__(self, model: str = "meta-llama/llama-4-scout-17b-16e-instruct", temperature: float = 0.85):
        """
        Initialize Groq provider.

        Args:
            model: Model name (default: meta-llama/llama-4-scout-17b-16e-instruct)
            temperature: Sampling temperature (0.0 to 2.0)
        """
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Groq not installed. Run: pip install groq")

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.conversation_history = []
        self.history_file = "conversation_history_groq.pkl"
        self.load_conversation_history()

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate response from Groq-hosted model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the API

        Returns:
            Generated response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=kwargs.get("max_tokens", 1024),
                top_p=kwargs.get("top_p", 1),
                stream=False,
                stop=kwargs.get("stop", None)
            )

            assistant_message = completion.choices[0].message.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            # Keep history limited
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # Save updated history
            self.save_conversation_history()

            return assistant_message

        except Exception as e:
            return f"Error generating response: {e}"

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """
        Generate streaming response from Groq-hosted model.

        Yields:
            Response chunks as they arrive
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=kwargs.get("max_tokens", 1024),
                top_p=kwargs.get("top_p", 1),
                stream=True,
                stop=kwargs.get("stop", None)
            )

            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

            # Update history after streaming completes
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": full_response})

            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # Save updated history
            self.save_conversation_history()

        except Exception as e:
            yield f"Error: {e}"

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.save_conversation_history()

    def save_conversation_history(self):
        """Save conversation history to file."""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.conversation_history, f)
        except Exception as e:
            print(f"Warning: Could not save conversation history: {e}")

    def load_conversation_history(self):
        """Load conversation history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    self.conversation_history = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load conversation history: {e}")
            self.conversation_history = []


class LLMManager:
    """Manager for multiple LLM providers."""
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize LLM manager.
        
        Args:
            provider: Provider name ("openai" or "anthropic")
            model: Optional model override
        """
        self.provider_name = provider.lower()
        
        if self.provider_name == "openai":
            self.provider = OpenAIProvider(model or "gpt-4o-mini")
        elif self.provider_name == "anthropic":
            self.provider = AnthropicProvider(model or "claude-sonnet-4-20250514")
        elif self.provider_name == "groq":
            self.provider = GroqProvider(model or "meta-llama/llama-4-scout-17b-16e-instruct")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using selected provider."""
        return self.provider.generate_response(prompt, system_prompt, **kwargs)
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate streaming response using selected provider."""
        return self.provider.generate_stream(prompt, system_prompt, **kwargs)
    
    def clear_history(self):
        """Clear conversation history."""
        self.provider.clear_history()

    def get_history_length(self):
        """Get the current length of conversation history."""
        return len(self.provider.conversation_history)
    
    def switch_provider(self, provider: str, model: Optional[str] = None):
        """Switch to different provider."""
        self.__init__(provider, model)


if __name__ == "__main__":
    # Example usage
    import sys
    
    print("Testing LLM Providers")
    print("-" * 50)
    
    # Test OpenAI
    try:
        print("Testing OpenAI Provider:")
        openai_provider = OpenAIProvider(model="gpt-4o-mini")
        response = openai_provider.generate_response(
            "What's the weather like today?",
            system_prompt="You are a helpful assistant. Keep responses brief."
        )
        print(f"OpenAI Response: {response}\n")
    except Exception as e:
        print(f"OpenAI Error: {e}\n")
    
    # Test Anthropic
    try:
        print("Testing Anthropic Provider:")
        anthropic_provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        response = anthropic_provider.generate_response(
            "What's the weather like today?",
            system_prompt="You are a helpful assistant. Keep responses brief."
        )
        print(f"Anthropic Response: {response}\n")
    except Exception as e:
        print(f"Anthropic Error: {e}\n")

    # Test Groq
    try:
        print("Testing Groq Provider:")
        groq_provider = GroqProvider()
        response = groq_provider.generate_response(
            "What's the weather like today?",
            system_prompt="You are a helpful assistant. Keep responses brief."
        )
        print(f"Groq Response: {response}\n")
    except Exception as e:
        print(f"Groq Error: {e}\n")
    
    # Test streaming
    try:
        print("Testing Streaming (Groq):")
        manager = LLMManager(provider="groq")
        print("Response: ", end="")
        for chunk in manager.generate_stream("Count to 5 slowly"):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Streaming Error: {e}\n")