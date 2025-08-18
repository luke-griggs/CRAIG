#!/usr/bin/env python3
"""
Voice Assistant with Wake Word Detection
Combines wake word detection, speech transcription, LLM processing, and TTS output.
"""

import os
import sys
import time
import threading
import signal
from typing import Optional
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Import our modules
from wake_word_detector import WakeWordDetector, OpenWakeWordDetector
from audio_transcriber import AudioTranscriber, SimpleAudioRecorder
from llm_providers import LLMManager
from tts_engine import TTSManager

# Initialize colorama for colored output
init(autoreset=True)

load_dotenv()

class VoiceAssistant:
    def __init__(self, 
                 wake_word: str = "jarvis",
                 use_porcupine: bool = True,
                 llm_provider: str = "openai",
                 llm_model: Optional[str] = None,
                 tts_engine: str = "pyttsx3",
                 recording_duration: float = 5.0,
                 use_streaming: bool = True):
        """
        Initialize Voice Assistant.
        
        Args:
            wake_word: Wake word to listen for
            use_porcupine: Use Porcupine (True) or OpenWakeWord (False)
            llm_provider: LLM provider ("openai" or "anthropic")
            llm_model: Optional model override
            tts_engine: TTS engine ("pyttsx3", "gtts", "system")
            recording_duration: Duration to record after wake word
            use_streaming: Use streaming transcription (True) or post-processing (False)
        """
        print(f"{Fore.CYAN}Initializing Voice Assistant...{Style.RESET_ALL}")
        
        self.wake_word = wake_word
        self.recording_duration = recording_duration
        self.use_streaming = use_streaming
        self.is_listening = False
        self.is_processing = False
        
        # Initialize components
        self._init_wake_word_detector(use_porcupine)
        self._init_transcriber()
        self._init_llm(llm_provider, llm_model)
        self._init_tts(tts_engine)
        
        # System prompt for the assistant
        self.system_prompt = """You are a helpful voice assistant. 
        Keep your responses concise and natural for speech. 
        Avoid using markdown formatting or special characters.
        Be friendly and conversational."""
        
        print(f"{Fore.GREEN}✓ Voice Assistant initialized!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Wake word: '{wake_word}'{Style.RESET_ALL}")
    
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
        """Initialize audio transcriber."""
        try:
            if self.use_streaming:
                self.transcriber = AudioTranscriber(
                    on_transcript=self.on_partial_transcript,
                    on_final_transcript=self.on_final_transcript,
                    sample_rate=16000
                )
            else:
                self.transcriber = SimpleAudioRecorder(sample_rate=16000)
            print(f"{Fore.GREEN}✓ Transcriber ready{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ Transcriber error: {e}{Style.RESET_ALL}")
            # Fallback to simple recorder
            self.transcriber = SimpleAudioRecorder(sample_rate=16000)
            self.use_streaming = False
    
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
        if self.is_processing:
            return
        
        print(f"\n{Fore.GREEN}▶ Wake word detected!{Style.RESET_ALL}")
        
        # Play acknowledgment sound or speak
        self.tts.speak("Yes?", blocking=True)
        
        # Start recording/transcribing
        self.is_processing = True
        threading.Thread(target=self.process_voice_input).start()
    
    def on_partial_transcript(self, text: str):
        """Callback for partial transcripts (streaming only)."""
        print(f"{Fore.CYAN}〉 {text}{Style.RESET_ALL}", end="\r")
    
    def on_final_transcript(self, text: str):
        """Callback for final transcript."""
        print(f"\n{Fore.BLUE}User: {text}{Style.RESET_ALL}")
        self.last_transcript = text
    
    def process_voice_input(self):
        """Process voice input after wake word."""
        try:
            # Record and transcribe
            if self.use_streaming:
                print(f"{Fore.YELLOW}Listening...{Style.RESET_ALL}")
                self.transcriber.start_recording(duration=self.recording_duration)
                time.sleep(self.recording_duration + 1)
                transcript = self.transcriber.stop_recording()
            else:
                print(f"{Fore.YELLOW}Recording for {self.recording_duration}s...{Style.RESET_ALL}")
                audio_data = self.transcriber.record_audio(duration=self.recording_duration)
                print(f"{Fore.YELLOW}Transcribing...{Style.RESET_ALL}")
                transcript = self.transcriber.transcribe_audio(audio_data)
                print(f"{Fore.BLUE}User: {transcript}{Style.RESET_ALL}")
            
            if not transcript or transcript.strip() == "":
                print(f"{Fore.YELLOW}No speech detected{Style.RESET_ALL}")
                self.is_processing = False
                return
            
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
            print(f"{Fore.RED}Error processing voice input: {e}{Style.RESET_ALL}")
            self.tts.speak("Sorry, I encountered an error.", blocking=True)
        
        finally:
            self.is_processing = False
    
    def start(self):
        """Start the voice assistant."""
        print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Voice Assistant Started{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Say '{self.wake_word}' to activate{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to exit{Style.RESET_ALL}\n")
        
        self.is_listening = True
        self.wake_detector.start_listening()
        
        # Keep running
        try:
            while self.is_listening:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the voice assistant."""
        print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
        
        self.is_listening = False
        
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
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Recording duration in seconds")
    parser.add_argument("--no-streaming", action="store_true",
                       help="Disable streaming transcription")
    
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
        recording_duration=args.duration,
        use_streaming=not args.no_streaming
    )
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        assistant.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start assistant
    assistant.start()


if __name__ == "__main__":
    main()