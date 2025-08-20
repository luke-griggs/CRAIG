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
from wake_word_detector import WakeWordDetector, OpenWakeWordDetector
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
                 wake_word: str = "jarvis",
                 use_porcupine: bool = True,
                 llm_provider: str = "openai",
                 llm_model: Optional[str] = None,
                 tts_engine: str = "pyttsx3",
                 idle_timeout: float = 10.0):
        """
        Initialize Voice Assistant with conversation mode.
        
        Args:
            wake_word: Wake word to listen for
            use_porcupine: Use Porcupine (True) or OpenWakeWord (False)
            llm_provider: LLM provider ("openai" or "anthropic")
            llm_model: Optional model override
            tts_engine: TTS engine ("pyttsx3", "gtts", "system")
            idle_timeout: Seconds of silence before returning to wake word mode
        """
        print(f"{Fore.CYAN}Initializing Voice Assistant...{Style.RESET_ALL}")
        
        self.wake_word = wake_word
        self.idle_timeout = idle_timeout
        
        # State management
        self.mode = AssistantMode.WAKE_WORD
        self.is_running = False
        self.last_interaction = time.time()
        
        # Communication between threads
        self.transcript_queue = queue.Queue()
        
        # Initialize components
        self._init_wake_word_detector(use_porcupine)
        self._init_transcriber()
        self._init_llm(llm_provider, llm_model)
        self._init_tts(tts_engine)
        
        # Streaming thread (will be created when entering conversation mode)
        self.streaming_thread = None
        
        # System prompt for the assistant
        self.system_prompt = """You are JARVIS, a helpful AI assistant. 
        Keep your responses concise and natural for speech. 
        Avoid using markdown formatting or special characters.
        Be friendly and conversational."""
        
        print(f"{Fore.GREEN}✓ Voice Assistant initialized!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Wake word: '{wake_word}'{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Idle timeout: {idle_timeout}s{Style.RESET_ALL}")
    
    def _init_wake_word_detector(self, use_porcupine: bool):
        """Initialize wake word detector."""
        try:
            if use_porcupine:
                self.wake_detector = WakeWordDetector(
                    wake_word=self.wake_word,
                    sensitivity=0.5,
                    callback=self.on_wake_word_detected
                )
            else:
                # Use OpenWakeWord as fallback
                self.wake_detector = OpenWakeWordDetector(
                    model_path=self.wake_word if self.wake_word in ["alexa", "hey_mycroft"] else "alexa",
                    threshold=0.5,
                    callback=self.on_wake_word_detected
                )
            print(f"{Fore.GREEN}✓ Wake word detector ready{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ Wake word detector error: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Trying alternative detector...{Style.RESET_ALL}")
            # Fallback to OpenWakeWord
            self.wake_detector = OpenWakeWordDetector(
                model_path="alexa",
                threshold=0.5,
                callback=self.on_wake_word_detected
            )
    
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
    
    def _init_tts(self, engine: str):
        """Initialize TTS engine."""
        try:
            self.tts = TTSManager(engine=engine)
            print(f"{Fore.GREEN}✓ TTS engine ready ({engine}){Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ TTS error: {e}{Style.RESET_ALL}")
            # Fallback to system TTS
            self.tts = TTSManager(engine="system")
    
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
        
        # Stop wake word detection
        self.wake_detector.stop_listening()
        
        # Acknowledge activation
        print(f"{Fore.YELLOW}Entering conversation mode...{Style.RESET_ALL}")
        self.tts.speak("Yes?", blocking=True)
        
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
        
        # Say goodbye
        self.tts.speak("Going back to sleep", blocking=True)
        
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
        
        # Pause streaming during response
        self.transcriber.pause_streaming()
        
        try:
            # Get response from LLM
            print(f"{Fore.YELLOW}Thinking...{Style.RESET_ALL}")
            response = self.llm.generate_response(
                prompt=transcript,
                system_prompt=self.system_prompt,
                max_tokens=150  # Keep responses concise for speech
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
        print(f"Listening for wake word: '{self.wake_word}'...")
        
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
        print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
        
        self.is_running = False
        
        # Stop streaming if active
        if self.mode == AssistantMode.CONVERSATION:
            self.transcriber.stop_streaming()
        
        # Clean up components
        if self.wake_detector:
            self.wake_detector.cleanup()
        
        if self.transcriber:
            self.transcriber.cleanup()
        
        if self.tts:
            self.tts.stop()
        
        print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Assistant with Wake Word Detection")
    parser.add_argument("--wake-word", default="jarvis", 
                       help="Wake word (jarvis, alexa, hey_siri, ok_google, or path to .ppn file)")
    parser.add_argument("--no-porcupine", action="store_true",
                       help="Use OpenWakeWord instead of Porcupine")
    parser.add_argument("--llm", default="openai", choices=["openai", "anthropic"],
                       help="LLM provider")
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--tts", default="pyttsx3", choices=["pyttsx3", "gtts", "system"],
                       help="TTS engine")
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
    
    if not os.getenv("ASSEMBLYAI_API_KEY"):
        print(f"{Fore.RED}Error: ASSEMBLYAI_API_KEY not found in environment{Style.RESET_ALL}")
        print("Please set your AssemblyAI API key in .env file")
        sys.exit(1)
    
    if not args.no_porcupine and not os.getenv("PICOVOICE_ACCESS_KEY"):
        print(f"{Fore.YELLOW}Warning: PICOVOICE_ACCESS_KEY not found{Style.RESET_ALL}")
        print("Falling back to OpenWakeWord")
        args.no_porcupine = True
    
    # Create and start assistant
    assistant = VoiceAssistant(
        wake_word=args.wake_word,
        use_porcupine=not args.no_porcupine,
        llm_provider=args.llm,
        llm_model=args.model,
        tts_engine=args.tts,
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