import os
import requests
import io
import threading
import queue
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class ElevenLabsTTS:
    """Text-to-speech engine using ElevenLabs API."""
    
    def __init__(self, voice_id: str = "L0Dsvb3SLTyegXwtm47J"):
        """
        Initialize ElevenLabs TTS engine.
        
        Args:
            voice_id: ElevenLabs voice ID to use
        """
        self.api_key = os.getenv("ELEVEN_LABS_KEY")
        if not self.api_key:
            raise ValueError("ELEVEN_LABS_KEY not found in environment")
        
        self.voice_id = voice_id
        self.model_id = "eleven_flash_v2_5"
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        # Audio playback
        try:
            import pygame
            pygame.mixer.init()
            self.pygame = pygame
            self.use_pygame = True
        except ImportError:
            # Will use system commands for playback
            self.use_pygame = False
        
        # Speech queue for non-blocking speech
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = None
        self.current_stream = None
    
    def speak(self, text: str, blocking: bool = True):
        """
        Speak the given text using ElevenLabs API.
        
        Args:
            text: Text to speak
            blocking: If True, block until speech completes
        """
        if blocking:
            self._speak_text(text)
        else:
            self.speech_queue.put(text)
            if not self.is_speaking:
                self._start_speech_thread()
    
    def _speak_text(self, text: str):
        """Internal method to speak text."""
        try:
            # Prepare request
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                json=data,
                headers=headers,
                stream=True
            )
            
            if response.status_code != 200:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return
            
            # Play audio
            self._play_audio(response.content)
            
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
    
    def _play_audio(self, audio_data: bytes):
        """Play audio data."""
        import tempfile
        
        # Always use pygame for playback (simpler and more reliable)
        if not self.use_pygame:
            # Fallback: save to file and use system command
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                import platform
                import subprocess
                
                if platform.system() == "Darwin":  # macOS
                    subprocess.run(["afplay", tmp_file.name])
                elif platform.system() == "Linux":
                    subprocess.run(["mpg123", "-q", tmp_file.name])
                else:
                    print("Unsupported platform for audio playback")
                
                os.unlink(tmp_file.name)
        else:
            # Use pygame for playback
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                self.pygame.mixer.music.load(tmp_file.name)
                self.pygame.mixer.music.play()
                
                # Wait for playback to complete
                while self.pygame.mixer.music.get_busy():
                    self.pygame.time.Clock().tick(10)
                
                # Clean up
                os.unlink(tmp_file.name)
    
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
                self._speak_text(text)
                self.speech_queue.task_done()
            except queue.Empty:
                if self.speech_queue.empty():
                    self.is_speaking = False
            except Exception as e:
                print(f"Speech worker error: {e}")
    
    def stop(self):
        """Stop current speech."""
        self.is_speaking = False
        
        # Clear queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        
        # Stop current playback
        if self.use_pygame and self.pygame.mixer.music.get_busy():
            self.pygame.mixer.music.stop()
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()


class TTSManager:
    """Manager for TTS engine (ElevenLabs only)."""
    
    def __init__(self, voice_id: str = "L0Dsvb3SLTyegXwtm47J"):
        """
        Initialize TTS manager with ElevenLabs.
        
        Args:
            voice_id: ElevenLabs voice ID
        """
        self.engine = ElevenLabsTTS(voice_id=voice_id)
    
    def speak(self, text: str, blocking: bool = True):
        """Speak text using ElevenLabs."""
        self.engine.speak(text, blocking)
    
    def stop(self):
        """Stop current speech."""
        self.engine.stop()
    
    def cleanup(self):
        """Clean up resources."""
        self.engine.cleanup()


if __name__ == "__main__":
    # Test ElevenLabs TTS
    print("Testing ElevenLabs TTS Engine")
    print("-" * 50)
    
    try:
        tts = TTSManager()
        
        print("Speaking test message...")
        tts.speak("Hello! This is a test of the ElevenLabs text to speech engine using the Flash v2.5 model.", blocking=True)
        
        print("\nTesting non-blocking speech...")
        tts.speak("This is non-blocking speech.", blocking=False)
        print("This message prints while speech is playing")
        
        import time
        time.sleep(3)
        
        print("\nElevenLabs TTS test complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure ELEVEN_LABS_KEY is set in your .env file")