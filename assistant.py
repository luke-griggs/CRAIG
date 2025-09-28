#!/usr/bin/env python3
"""
Voice Assistant with Wake Word Detection
Implements a JARVIS-style assistant with conversation mode and idle timeout.
"""

import os
import sys
import time
import threading
import queue
import signal
from typing import Optional
from enum import Enum
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Import our modules
from wake_word_detector import WakeWordDetector
from audio_transcriber import AudioTranscriber
from llm_providers import LLMManager
from tts_engine import TTSManager

# Initialize colorama for colored output
init(autoreset=True)

load_dotenv()

class AssistantMode(Enum):
    WAKE_WORD = "wake_word"
    CONVERSATION = "conversation"

class VoiceAssistant:
    def __init__(self, 
                 wake_word: str = "craig",
                 use_porcupine: bool = True,
                 llm_provider: str = "openai",
                 llm_model: Optional[str] = None,
                 voice_id: str = "L0Dsvb3SLTyegXwtm47J",
                 idle_timeout: float = 10.0):
        """
        Initialize Voice Assistant with conversation mode.
        
        Args:
            llm_provider: LLM provider ("openai" or "anthropic")
            llm_model: Optional model override
            voice_id: ElevenLabs voice ID
            idle_timeout: Seconds of silence before returning to wake word mode
        """
        print(f"{Fore.CYAN}Initializing Voice Assistant...{Style.RESET_ALL}")
        
        self.idle_timeout = idle_timeout
        self.wake_word = wake_word
        
        # State management
        self.mode = AssistantMode.WAKE_WORD
        self.is_running = False
        self.last_interaction = time.time()
        
        # Communication between threads
        self.transcript_queue = queue.Queue()
        
        # Initialize components
        self._init_wake_word_detector()
        self._init_transcriber()
        self._init_llm(llm_provider, llm_model)
        self._init_tts(voice_id)
        
        # Streaming thread (will be created when entering conversation mode)
        self.streaming_thread = None
        
        # System prompt for the assistant
        self.system_prompt = f"""
        You are Craig, a slightly witty, subtly sarcastic voice assistant. Your style is like that lovable side character in a movie who's sharp, loyal, and funny without trying too hard. 
        the current time is {time.localtime()} 

        Tone & Style  
        - Speak like a clever friend, not like a corporate assistant.  
        - Use dry humor, observational wit, and light sarcasm - occasionally swearing.  
        - Keep responses concise, natural, and conversational — no filler like "I'm ready to tackle whatever you've got cookin'."  
        - Don't ramble when it's not needed(1-2 sentences unless the conversation calls for more)

        Behavior Rules  
        - Prioritize sounding real and personable over being "helpful."  
        - Don't constantly ask the user what they need — assume they'll tell you.
        
        Tool Usage:

        Rainbow Text Tool:
        - You have access to a rainbow text tool that displays colorful text in the console.
        - Use this tool when the user asks for something colorful, fun, or when you want to add visual flair.

        Weather Lookup Tool:
        - You can call a weather lookup tool to fetch the current conditions for a city (it needs the location).
        - If the user asks about the weather but omits a location, ask them where they mean before calling the tool.

        SMS (Twilio) Tool:
        - You can send SMS messages with the Twilio tool when the user explicitly asks you to send them something.
        """
        
        # Load available tools
        from tools import AVAILABLE_TOOLS
        self.tools = AVAILABLE_TOOLS
        
        print(f"{Fore.GREEN}✓ Voice Assistant initialized!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Idle timeout: {idle_timeout}s{Style.RESET_ALL}")
    
    def _init_wake_word_detector(self):
        """Initialize wake word detector with custom model support."""
        try:
            # Import the factory function
            from wake_word_detector import WakeWordDetector
            
            # Use factory function to get appropriate detector
            self.wake_detector = WakeWordDetector(
                model_path="models/craig.tflite",
                scaler_path="models/craig_scaler.pkl",
                threshold=0.520,
                callback=self.on_wake_word_detected
            )
            
            print(f"{Fore.GREEN}✓ Wake word detector ready{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Wake word detector error: {e}{Style.RESET_ALL}")
            raise
    
    def _init_transcriber(self):
        """Initialize audio transcriber with queue-based communication."""
        try:
            self.transcriber = AudioTranscriber(
                transcript_queue=self.transcript_queue,
                sample_rate=16000
            )
            print(f"{Fore.GREEN}✓ Transcriber ready{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ Transcriber error: {e}{Style.RESET_ALL}")
            raise
    
    def _init_llm(self, provider: str, model: Optional[str]):
        """Initialize LLM provider."""
        try:
            self.llm = LLMManager(provider=provider, model=model)
            print(f"{Fore.GREEN}✓ LLM provider ready ({provider}){Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ LLM error: {e}{Style.RESET_ALL}")
            raise
    
    def _init_tts(self, voice_id: str):
        """Initialize TTS engine."""
        try:
            self.tts = TTSManager(voice_id=voice_id)
            print(f"{Fore.GREEN}✓ TTS engine ready (ElevenLabs){Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ TTS error: {e}{Style.RESET_ALL}")
            raise
    
    def on_wake_word_detected(self):
        """Callback when wake word is detected."""
        if self.mode != AssistantMode.WAKE_WORD:
            return  # Ignore if already in conversation
        
        # Immediately switch mode to prevent duplicate detections
        self.mode = AssistantMode.CONVERSATION
        
        print(f"\n{Fore.GREEN}▶ Wake word detected!{Style.RESET_ALL}")
        
        # Enter conversation mode
        self.enter_conversation_mode()
    
    def enter_conversation_mode(self):
        """Enter conversation mode and start streaming."""
        # Mode already set in on_wake_word_detected
        self.last_interaction = time.time()
        
        # Conversation history persists across wake/sleep cycles
        
        # Stop wake word detection
        self.wake_detector.stop_listening()
        
        # Acknowledge activation
        print(f"{Fore.YELLOW}Entering conversation mode...{Style.RESET_ALL}")
        self.tts.speak("What the hell do you want?", blocking=True)
        
        # Start streaming transcription
        self.transcriber.start_streaming()
        
        print(f"{Fore.CYAN}Listening... (silence for {self.idle_timeout}s to exit){Style.RESET_ALL}")
    
    def exit_conversation_mode(self):
        """Exit conversation mode and return to wake word detection."""
        print(f"\n{Fore.YELLOW}Exiting conversation mode...{Style.RESET_ALL}")
        
        # Stop streaming
        self.transcriber.stop_streaming()
        
        # Clear any pending transcripts
        while not self.transcript_queue.empty():
            try:
                self.transcript_queue.get_nowait()
            except queue.Empty:
                break
        
        # Conversation history is preserved across wake/sleep cycles
        print(f"{Fore.YELLOW}Conversation mode exited, history preserved{Style.RESET_ALL}")
        
        # Return to wake word mode
        self.mode = AssistantMode.WAKE_WORD
        
        # Restart wake word detection
        self.wake_detector.start_listening()
        
        print(f"{Fore.CYAN}Listening for wake word: '{self.wake_word}'...{Style.RESET_ALL}")
    
    def process_transcript(self, transcript: str):
        """Process a transcript and generate response."""
        print(f"\n{Fore.BLUE}User: {transcript}{Style.RESET_ALL}")
        
        # Check for exit commands
        if transcript.lower() in ["goodbye", "bye", "exit", "stop", "go to sleep"]:
            self.exit_conversation_mode()
            return

        # Check for clear history commands
        if transcript.lower() in ["clear history", "forget everything", "reset memory", "wipe memory"]:
            self.llm.clear_history()
            self.tts.speak("Conversation history cleared. Starting fresh!", blocking=True)
            return

        # Check for history status commands
        if transcript.lower() in ["how many conversations", "conversation count", "history length"]:
            history_length = self.llm.get_history_length()
            if history_length == 0:
                self.tts.speak("No conversation history yet.", blocking=True)
            else:
                self.tts.speak(f"We've had {history_length // 2} exchanges in our conversation.", blocking=True)
            return
        
        # Pause streaming during response
        self.transcriber.pause_streaming()
        
        try:
            # Get response from LLM
            print(f"{Fore.YELLOW}Thinking...{Style.RESET_ALL}")
            response = self.llm.generate_response(
                prompt=transcript,
                system_prompt=self.system_prompt,
                tools=self.tools,  # Pass tools to the LLM
            )
            
            print(f"{Fore.GREEN}Assistant: {response}{Style.RESET_ALL}")
            
            # Speak response
            self.tts.speak(response, blocking=True)
            
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {e}{Style.RESET_ALL}")
            self.tts.speak("Sorry, I encountered an error.", blocking=True)
        
        finally:
            # Resume streaming after speaking
            self.transcriber.resume_streaming()
            
            # Update last interaction time
            self.last_interaction = time.time()
    
    def run(self):
        """Main run loop."""
        print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Voice Assistant Started{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Say '{self.wake_word}' to activate{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to exit{Style.RESET_ALL}\n")
        
        self.is_running = True
        
        # Start in wake word mode
        self.wake_detector.start_listening()
        
        try:
            while self.is_running:
                if self.mode == AssistantMode.WAKE_WORD:
                    # Just sleep, wake word detector handles detection
                    time.sleep(0.5)
                    
                elif self.mode == AssistantMode.CONVERSATION:
                    try:
                        # Wait for transcript with timeout
                        transcript = self.transcript_queue.get(timeout=1.0)
                        
                        # Process the transcript
                        self.process_transcript(transcript)
                        
                    except queue.Empty:
                        # No transcript received, check for idle timeout
                        if time.time() - self.last_interaction > self.idle_timeout:
                            print(f"\n{Fore.YELLOW}Idle timeout reached{Style.RESET_ALL}")
                            self.exit_conversation_mode()
                
        except KeyboardInterrupt:
            pass
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the voice assistant."""
        if not self.is_running:
            return  # Already stopped
            
        print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
        
        self.is_running = False
        
        # Stop streaming if active
        if self.mode == AssistantMode.CONVERSATION:
            try:
                self.transcriber.stop_streaming()
            except Exception as e:
                print(f"Error stopping transcriber: {e}")
        
        # Clean up components
        try:
            if self.wake_detector:
                self.wake_detector.cleanup()
                self.wake_detector = None
        except Exception as e:
            print(f"Error cleaning up wake detector: {e}")
        
        try:
            if self.transcriber:
                self.transcriber.cleanup()
                self.transcriber = None
        except Exception as e:
            print(f"Error cleaning up transcriber: {e}")
        
        try:
            if self.tts:
                self.tts.cleanup()
                self.tts = None
        except Exception as e:
            print(f"Error stopping TTS: {e}")
        
        print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Assistant with Wake Word Detection")

    parser.add_argument("--llm", default="groq", choices=["openai", "anthropic", "groq"],
                       help="LLM provider")
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--voice", default="2BJW5coyhAzSr8STdHbE",
                       help="ElevenLabs voice ID")
    parser.add_argument("--timeout", type=float, default=10.0,
                       help="Idle timeout in seconds")
    
    args = parser.parse_args()
    
    # Check for required API keys
    if args.llm == "openai" and not os.getenv("OPENAI_API_KEY"):
        print(f"{Fore.RED}Error: OPENAI_API_KEY not found in environment{Style.RESET_ALL}")
        print("Please set your OpenAI API key in .env file")
        sys.exit(1)

    if args.llm == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{Fore.RED}Error: ANTHROPIC_API_KEY not found in environment{Style.RESET_ALL}")
        print("Please set your Anthropic API key in .env file")
        sys.exit(1)

    if args.llm == "groq" and not os.getenv("GROQ_API_KEY"):
        print(f"{Fore.RED}Error: GROQ_API_KEY not found in environment{Style.RESET_ALL}")
        print("Please set your Groq API key in .env file")
        sys.exit(1)
    
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        print(f"{Fore.RED}Error: ASSEMBLYAI_API_KEY not found in environment{Style.RESET_ALL}")
        print("Please set your AssemblyAI API key in .env file")
        sys.exit(1)
    
    if not os.getenv("ELEVEN_LABS_KEY"):
        print(f"{Fore.RED}Error: ELEVEN_LABS_KEY not found in environment{Style.RESET_ALL}")
        print("Please set your ElevenLabs API key in .env file")
        sys.exit(1)
    
    # Create and start assistant
    assistant = VoiceAssistant(
        wake_word="craig",
        llm_provider=args.llm,
        llm_model=args.model,
        voice_id=args.voice,
        idle_timeout=args.timeout
    )
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        assistant.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start assistant
    assistant.run()


if __name__ == "__main__":
    main()
