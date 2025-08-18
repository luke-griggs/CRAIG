import os
import threading
import queue
from typing import Optional
from abc import ABC, abstractmethod
import platform

class TTSEngine(ABC):
    """Abstract base class for TTS engines."""
    
    @abstractmethod
    def speak(self, text: str):
        """Speak the given text."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop current speech."""
        pass
    
    @abstractmethod
    def set_voice(self, voice_id: str):
        """Set the voice to use."""
        pass
    
    @abstractmethod
    def set_rate(self, rate: int):
        """Set speech rate."""
        pass


class Pyttsx3Engine(TTSEngine):
    """Text-to-speech engine using pyttsx3 (offline)."""
    
    def __init__(self, voice_index: int = 0, rate: int = 180):
        """
        Initialize pyttsx3 TTS engine.
        
        Args:
            voice_index: Index of voice to use
            rate: Speech rate (words per minute)
        """
        try:
            import pyttsx3
        except ImportError:
            raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3")
        
        self.engine = pyttsx3.init()
        self.rate = rate
        self.voice_index = voice_index
        
        # Configure engine
        self._configure_engine()
        
        # Speech queue for non-blocking speech
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = None
    
    def _configure_engine(self):
        """Configure TTS engine settings."""
        # Set rate
        self.engine.setProperty('rate', self.rate)
        
        # Set volume (0.0 to 1.0)
        self.engine.setProperty('volume', 1.0)
        
        # Set voice
        voices = self.engine.getProperty('voices')
        if voices and self.voice_index < len(voices):
            self.engine.setProperty('voice', voices[self.voice_index].id)
            print(f"Using voice: {voices[self.voice_index].name}")
    
    def speak(self, text: str, blocking: bool = False):
        """
        Speak the given text.
        
        Args:
            text: Text to speak
            blocking: If True, block until speech completes
        """
        if blocking:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            self.speech_queue.put(text)
            if not self.is_speaking:
                self._start_speech_thread()
    
    def _start_speech_thread(self):
        """Start background thread for speech."""
        if self.speech_thread and self.speech_thread.is_alive():
            return
        
        self.is_speaking = True
        self.speech_thread = threading.Thread(target=self._speech_worker)
        self.speech_thread.daemon = True
        self.speech_thread.start()
    
    def _speech_worker(self):
        """Worker thread for processing speech queue."""
        while self.is_speaking:
            try:
                text = self.speech_queue.get(timeout=1)
                self.engine.say(text)
                self.engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                if self.speech_queue.empty():
                    self.is_speaking = False
            except Exception as e:
                print(f"Speech error: {e}")
    
    def stop(self):
        """Stop current speech."""
        self.is_speaking = False
        # Clear queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        self.engine.stop()
    
    def set_voice(self, voice_index: int):
        """Set voice by index."""
        voices = self.engine.getProperty('voices')
        if voices and voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
            self.voice_index = voice_index
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)."""
        self.engine.setProperty('rate', rate)
        self.rate = rate
    
    def list_voices(self):
        """List available voices."""
        voices = self.engine.getProperty('voices')
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name} - {voice.id}")
            if voice.languages:
                print(f"   Languages: {voice.languages}")


class GTTSEngine(TTSEngine):
    """Text-to-speech engine using Google TTS (requires internet)."""
    
    def __init__(self, language: str = 'en', slow: bool = False):
        """
        Initialize gTTS engine.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
            slow: If True, speak slowly
        """
        try:
            from gtts import gTTS
            import pygame
        except ImportError:
            raise ImportError("gTTS/pygame not installed. Run: pip install gtts pygame")
        
        self.gTTS = gTTS
        self.language = language
        self.slow = slow
        
        # Initialize pygame mixer for audio playback
        import pygame
        pygame.mixer.init()
        self.pygame = pygame
        
        self.is_playing = False
        self.temp_files = []
    
    def speak(self, text: str, blocking: bool = True):
        """
        Speak the given text using Google TTS.
        
        Args:
            text: Text to speak
            blocking: If True, block until speech completes
        """
        try:
            # Create gTTS object
            tts = self.gTTS(text=text, lang=self.language, slow=self.slow)
            
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tts.save(tmp_file.name)
                self.temp_files.append(tmp_file.name)
                
                # Play audio
                self.pygame.mixer.music.load(tmp_file.name)
                self.pygame.mixer.music.play()
                self.is_playing = True
                
                if blocking:
                    # Wait for playback to complete
                    while self.pygame.mixer.music.get_busy():
                        self.pygame.time.Clock().tick(10)
                    self.is_playing = False
                    
                    # Clean up temp file
                    os.unlink(tmp_file.name)
                    self.temp_files.remove(tmp_file.name)
                    
        except Exception as e:
            print(f"gTTS error: {e}")
    
    def stop(self):
        """Stop current speech."""
        if self.is_playing:
            self.pygame.mixer.music.stop()
            self.is_playing = False
        
        # Clean up temp files
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass
        self.temp_files.clear()
    
    def set_voice(self, voice_id: str):
        """gTTS doesn't support voice selection."""
        print("gTTS doesn't support voice selection")
    
    def set_rate(self, rate: int):
        """Set speech rate (only slow/normal supported)."""
        self.slow = rate < 150


class SystemTTSEngine(TTSEngine):
    """System TTS using OS commands (macOS/Linux)."""
    
    def __init__(self):
        """Initialize system TTS engine."""
        self.system = platform.system()
        self.current_process = None
        
        if self.system == "Darwin":  # macOS
            self.command = "say"
            self.voice = None
            self.rate = 180
        elif self.system == "Linux":
            # Check for espeak or festival
            import subprocess
            try:
                subprocess.run(["which", "espeak"], check=True, capture_output=True)
                self.command = "espeak"
            except:
                try:
                    subprocess.run(["which", "festival"], check=True, capture_output=True)
                    self.command = "festival"
                except:
                    raise RuntimeError("No TTS command found. Install espeak or festival.")
        else:
            raise RuntimeError(f"System TTS not supported on {self.system}")
    
    def speak(self, text: str, blocking: bool = True):
        """
        Speak text using system command.
        
        Args:
            text: Text to speak
            blocking: If True, block until speech completes
        """
        import subprocess
        
        # Escape text for shell
        text = text.replace('"', '\\"').replace("'", "\\'")
        
        if self.system == "Darwin":  # macOS
            cmd = [self.command]
            if self.voice:
                cmd.extend(["-v", self.voice])
            if self.rate:
                cmd.extend(["-r", str(self.rate)])
            cmd.append(text)
        elif self.command == "espeak":
            cmd = [self.command, text]
        elif self.command == "festival":
            cmd = ["echo", text, "|", "festival", "--tts"]
        
        if blocking:
            subprocess.run(cmd, shell=(self.command == "festival"))
        else:
            self.current_process = subprocess.Popen(
                cmd, 
                shell=(self.command == "festival"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    
    def stop(self):
        """Stop current speech."""
        if self.current_process:
            self.current_process.terminate()
            self.current_process = None
    
    def set_voice(self, voice_id: str):
        """Set voice (macOS only)."""
        if self.system == "Darwin":
            self.voice = voice_id
    
    def set_rate(self, rate: int):
        """Set speech rate (macOS only)."""
        if self.system == "Darwin":
            self.rate = rate
    
    def list_voices(self):
        """List available voices (macOS only)."""
        if self.system == "Darwin":
            import subprocess
            result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
            print("Available voices:")
            print(result.stdout)


class TTSManager:
    """Manager for TTS engines."""
    
    def __init__(self, engine: str = "pyttsx3"):
        """
        Initialize TTS manager.
        
        Args:
            engine: Engine to use ("pyttsx3", "gtts", "system")
        """
        self.engine_name = engine.lower()
        
        if self.engine_name == "pyttsx3":
            self.engine = Pyttsx3Engine()
        elif self.engine_name == "gtts":
            self.engine = GTTSEngine()
        elif self.engine_name == "system":
            self.engine = SystemTTSEngine()
        else:
            # Default to pyttsx3
            self.engine = Pyttsx3Engine()
    
    def speak(self, text: str, blocking: bool = False):
        """Speak text using selected engine."""
        self.engine.speak(text, blocking)
    
    def stop(self):
        """Stop current speech."""
        self.engine.stop()
    
    def set_voice(self, voice_id):
        """Set voice."""
        self.engine.set_voice(voice_id)
    
    def set_rate(self, rate: int):
        """Set speech rate."""
        self.engine.set_rate(rate)


if __name__ == "__main__":
    # Test TTS engines
    import time
    
    print("Testing TTS Engines")
    print("-" * 50)
    
    # Test pyttsx3
    try:
        print("Testing pyttsx3...")
        tts = Pyttsx3Engine()
        tts.list_voices()
        tts.speak("Hello, this is a test of the text to speech engine.", blocking=True)
        print("pyttsx3 test complete\n")
    except Exception as e:
        print(f"pyttsx3 error: {e}\n")
    
    # Test system TTS (macOS/Linux)
    try:
        print("Testing system TTS...")
        tts = SystemTTSEngine()
        tts.speak("Testing system text to speech.", blocking=True)
        print("System TTS test complete\n")
    except Exception as e:
        print(f"System TTS error: {e}\n")
    
    # Test gTTS (requires internet)
    try:
        print("Testing gTTS (requires internet)...")
        tts = GTTSEngine()
        tts.speak("Testing Google text to speech.", blocking=True)
        print("gTTS test complete\n")
    except Exception as e:
        print(f"gTTS error: {e}\n")
    
    # Test manager
    print("Testing TTS Manager...")
    manager = TTSManager(engine="pyttsx3")
    manager.speak("Voice assistant ready!", blocking=True)
    print("All tests complete!")