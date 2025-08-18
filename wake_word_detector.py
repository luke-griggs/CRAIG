import os
import struct
import pyaudio
import pvporcupine
from typing import Optional, Callable
from dotenv import load_dotenv
import threading
import queue

load_dotenv()

class WakeWordDetector:
    def __init__(self, 
                 wake_word: str = "jarvis",
                 sensitivity: float = 0.5,
                 callback: Optional[Callable] = None):
        """
        Initialize wake word detector using Picovoice Porcupine.
        
        Args:
            wake_word: Built-in wake word or path to custom .ppn file
            sensitivity: Detection sensitivity (0.0 to 1.0)
            callback: Function to call when wake word is detected
        """
        self.access_key = os.getenv("PICOVOICE_ACCESS_KEY")
        if not self.access_key:
            raise ValueError("PICOVOICE_ACCESS_KEY not found in environment")
        
        self.wake_word = wake_word
        self.sensitivity = sensitivity
        self.callback = callback
        self.is_listening = False
        self._audio_queue = queue.Queue()
        
        # Initialize Porcupine
        try:
            keyword_paths = None
            keywords = None
            
            # Check if using custom wake word file
            if wake_word.endswith('.ppn'):
                keyword_paths = [wake_word]
            else:
                # Use built-in wake words
                keywords = [wake_word]
            
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=keyword_paths,
                keywords=keywords,
                sensitivities=[sensitivity]
            )
            
            self.pa = pyaudio.PyAudio()
            self.audio_stream = None
            
        except Exception as e:
            print(f"Error initializing Porcupine: {e}")
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for continuous audio streaming."""
        if in_data:
            self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def start_listening(self):
        """Start listening for wake word."""
        if self.is_listening:
            return
        
        self.is_listening = True
        
        # Open audio stream
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
            stream_callback=self._audio_callback
        )
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        print(f"Listening for wake word: '{self.wake_word}'...")
    
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        while self.is_listening:
            try:
                # Get audio data from queue
                audio_data = self._audio_queue.get(timeout=1)
                
                # Convert bytes to int16 array
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, audio_data)
                
                # Process audio frame
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    print(f"Wake word detected!")
                    if self.callback:
                        self.callback()
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in detection loop: {e}")
    
    def stop_listening(self):
        """Stop listening for wake word."""
        self.is_listening = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        # Wait for detection thread to finish
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join(timeout=2)
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_listening()
        
        if self.porcupine:
            self.porcupine.delete()
        
        if self.pa:
            self.pa.terminate()


class OpenWakeWordDetector:
    """Alternative wake word detector using OpenWakeWord (no API key required)."""
    
    def __init__(self, 
                 model_path: str = "alexa",
                 threshold: float = 0.5,
                 callback: Optional[Callable] = None):
        """
        Initialize OpenWakeWord detector.
        
        Args:
            model_path: Pre-trained model name or path to custom model
            threshold: Detection threshold (0.0 to 1.0)
            callback: Function to call when wake word is detected
        """
        try:
            import openwakeword
            from openwakeword.model import Model
            
            self.model = Model(
                wakeword_models=[model_path],
                inference_framework="onnx"
            )
            
            self.threshold = threshold
            self.callback = callback
            self.is_listening = False
            
            self.pa = pyaudio.PyAudio()
            self.audio_stream = None
            
        except ImportError:
            raise ImportError("OpenWakeWord not installed. Run: pip install openwakeword")
    
    def start_listening(self):
        """Start listening for wake word."""
        if self.is_listening:
            return
        
        self.is_listening = True
        
        # Open audio stream
        self.audio_stream = self.pa.open(
            rate=16000,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1280,  # 80ms chunks at 16kHz
            stream_callback=self._audio_callback
        )
        
        print(f"Listening for wake word...")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio and detect wake word."""
        if in_data and self.is_listening:
            # Convert bytes to numpy array
            import numpy as np
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Get predictions
            prediction = self.model.predict(audio_data)
            
            # Check if wake word detected
            for model_name, score in prediction.items():
                if score > self.threshold:
                    print(f"Wake word detected: {model_name} (score: {score:.2f})")
                    if self.callback:
                        self.callback()
        
        return (None, pyaudio.paContinue)
    
    def stop_listening(self):
        """Stop listening for wake word."""
        self.is_listening = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_listening()
        
        if self.pa:
            self.pa.terminate()


if __name__ == "__main__":
    # Example usage
    def on_wake_word_detected():
        print("Wake word detected! Ready to listen...")
    
    # Using Porcupine (requires API key)
    try:
        detector = WakeWordDetector(
            wake_word="jarvis",  # or "alexa", "hey_siri", "ok_google"
            sensitivity=0.5,
            callback=on_wake_word_detected
        )
        detector.start_listening()
        
        # Keep running
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        detector.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to OpenWakeWord...")
        
        # Fallback to OpenWakeWord (no API key required)
        detector = OpenWakeWordDetector(
            model_path="alexa",
            threshold=0.5,
            callback=on_wake_word_detected
        )
        detector.start_listening()
        
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            detector.cleanup()