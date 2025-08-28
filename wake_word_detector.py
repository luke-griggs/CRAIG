import os
import struct
import pyaudio
import pvporcupine
from typing import Optional, Callable
from dotenv import load_dotenv
import threading
import queue
from collections import deque
import numpy as np
import tensorflow as tf
import joblib

load_dotenv()

class WakeWordDetector:
    """Custom wake word detector for user-trained TensorFlow Lite models."""
    
    def __init__(self, model_path: str, scaler_path: str, threshold: float = 0.6, callback=None):
        """
        Initialize custom wake word detector.
        
        Args:
            model_path: Path to your trained .tflite model file
            scaler_path: Path to your saved scaler (.pkl file)  
            threshold: Detection threshold (0.0 to 1.0)
            callback: Function to call when wake word is detected
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.threshold = threshold
        self.callback = callback
        
        # Audio parameters (MUST match training!)
        self.sample_rate = 16000
        self.chunk_size = 512  # PyAudio frame size
        self.window_duration = 1.0  # 1 second windows (same as training)
        self.window_samples = int(self.sample_rate * self.window_duration)  # 16,000 samples
        
        # Rolling buffer to accumulate audio
        self.audio_buffer = deque(maxlen=self.window_samples)
        self.is_listening = False
        
        # Thread lock for TFLite inference
        self.inference_lock = threading.Lock()
        
        # Load trained model and scaler
        self._load_model()
        self._load_scaler()
        
        # PyAudio setup
        self.pa = pyaudio.PyAudio()
        self.audio_stream = None
        
        print(f"Custom wake word detector initialized (threshold: {threshold})")
    
    def _load_model(self):
        """Load TensorFlow Lite model."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"TFLite model loaded: {self.model_path}")
            print(f"  Input shape: {self.input_details[0]['shape']}")
            print(f"  Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model {self.model_path}: {e}")
    
    def _load_scaler(self):
        """Load the feature scaler used during training."""
        try:
            self.scaler = joblib.load(self.scaler_path)
            print(f"Feature scaler loaded: {self.scaler_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler {self.scaler_path}: {e}")
    
    def audio_bytes_to_waveform(self, audio_bytes):
        """
        Convert PyAudio bytes to numpy waveform (same format as librosa.load).
        """
        # Convert bytes to 16-bit integers
        audio_ints = struct.unpack(f"{len(audio_bytes)//2}h", audio_bytes)
        
        # Convert to float32 and normalize (same as librosa.load does)
        audio_waveform = np.array(audio_ints, dtype=np.float32)
        audio_waveform = audio_waveform / 32768.0  # Normalize int16 to [-1, 1]
        
        return audio_waveform
    
    def extract_mfcc_features(self, audio_waveform):
        """
        Extract MFCC features from waveform.
        MUST be identical to training preprocessing!
        """
        import librosa
        
        # Ensure we have exactly the right length
        if len(audio_waveform) < self.window_samples:
            # Pad with zeros if too short
            audio_waveform = np.pad(audio_waveform, 
                                  (0, self.window_samples - len(audio_waveform)))
        elif len(audio_waveform) > self.window_samples:
            # Trim if too long
            audio_waveform = audio_waveform[:self.window_samples]
        
        # Extract MFCC features (EXACT same parameters as training!)
        mfcc_matrix = librosa.feature.mfcc(
            y=audio_waveform,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=512,
            hop_length=160
        )
        
        # Calculate statistics (same as training!)
        features = np.concatenate([
            np.mean(mfcc_matrix, axis=1),  # 13 mean values
            np.std(mfcc_matrix, axis=1),   # 13 std values
            np.max(mfcc_matrix, axis=1),   # 13 max values
            np.min(mfcc_matrix, axis=1)    # 13 min values
        ])  # Total: 52 features
        
        return features
    
    def _predict(self, features):
        """Run TensorFlow Lite inference with thread safety."""
        with self.inference_lock:  # Prevent concurrent inference calls
            # Apply same scaling as training
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            features_scaled = features_scaled.astype(np.float32).copy()  # Ensure we own the data
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], features_scaled)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output (probability) and immediately copy to avoid reference issues
            prediction = self.interpreter.get_tensor(self.output_details[0]['index']).copy()
            return float(prediction[0, 0])  # Convert to Python float to break reference
    
    def start_listening(self):
        """Start listening for wake word."""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.audio_buffer.clear()
        
        # Open audio stream
        self.audio_stream = self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        print("Listening for custom wake word...")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio and detect wake word."""
        if in_data and self.is_listening:
            # Convert bytes to waveform
            audio_chunk = self.audio_bytes_to_waveform(in_data)
            
            # Add to rolling buffer
            self.audio_buffer.extend(audio_chunk)
            
            # Only process when we have enough audio (1 second)
            if len(self.audio_buffer) >= self.window_samples:
                # Get last 1 second of audio
                audio_window = np.array(list(self.audio_buffer)[-self.window_samples:])
                
                # Process in separate thread to avoid blocking audio
                threading.Thread(
                    target=self._process_audio_window, 
                    args=(audio_window.copy(),),
                    daemon=True
                ).start()
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_window(self, audio_window):
        """Process 1-second audio window for wake word detection."""
        try:
            # Extract features (same pipeline as training!)
            features = self.extract_mfcc_features(audio_window)
            
            # Run inference
            confidence = self._predict(features)
            
            # Debug output (remove this once working)
            print(f"Wake word confidence: {confidence:.3f}")
            
            # Check threshold
            if confidence > self.threshold:
                print(f"Custom wake word detected! Confidence: {confidence:.3f}")
                if self.callback:
                    self.callback()
                
                # Clear buffer to avoid multiple detections
                self.audio_buffer.clear()
                
        except Exception as e:
            print(f"Error processing audio window: {e}")
    
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
    # Test your custom Craig detector
    def on_wake_word_detected():
        print("Craig detected! Voice assistant activated.")
    
    # Test Craig detector specifically
    try:
        print("Testing Craig detector...")
        detector = WakeWordDetector(
            wake_word="craig",
            callback=on_wake_word_detected
        )
        
        detector.start_listening()
        print("Say 'Craig' to test...")
        
        # Keep running
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        detector.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("  - models/craig.tflite")
        print("  - models/craig_scaler.pkl")
        print("  - Required dependencies installed")