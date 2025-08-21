import os
import pyaudio
import threading
import queue
import time
from typing import Optional, Callable
from dotenv import load_dotenv
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)

load_dotenv()

class AudioTranscriber:
    def __init__(self, 
                 transcript_queue: Optional[queue.Queue] = None,
                 sample_rate: int = 16000):
        """
        Initialize real-time audio transcriber with AssemblyAI.
        
        Args:
            transcript_queue: Queue to send complete transcripts to
            sample_rate: Audio sample rate (16000 or higher recommended)
        """
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment")
        
        aai.settings.api_key = self.api_key
        
        self.transcript_queue = transcript_queue or queue.Queue()
        self.sample_rate = sample_rate
        
        # State management
        self.is_streaming = False
        self.is_paused = False
        
        # Audio setup
        self.pa = pyaudio.PyAudio()
        
        # Streaming client
        self.streaming_client = None
        self.streaming_thread = None
        self.session_id = None
        
        # Buffer for partial transcripts
        self.partial_buffer = []
        
    def _create_streaming_client(self):
        """Create AssemblyAI streaming client with event handlers."""
        
        def on_begin(client: StreamingClient, event: BeginEvent):
            self.session_id = event.id
            print(f"✓ Transcription session started")
        
        def on_turn(client: StreamingClient, event: TurnEvent):
            """Handle turn events (complete utterances)."""
            transcript = event.transcript
            
            # Skip if paused or no transcript
            if self.is_paused or not transcript:
                return
            
            # Check if this is the end of a turn (complete utterance)
            if hasattr(event, 'end_of_turn') and event.end_of_turn:
                # Send complete transcript to queue
                if self.transcript_queue:
                    self.transcript_queue.put(transcript)
                print(f"[Turn Complete] {transcript}")
            else:
                # Show partial transcript
                print(f"[Partial] {transcript}", end="\r")
        
        def on_error(client: StreamingClient, error: StreamingError):
            print(f"❌ Transcription error: {error}")
        
        def on_termination(client: StreamingClient, event: TerminationEvent):
            print(f"✓ Session terminated")
        
        # Create client with event handlers
        options = StreamingClientOptions(
            api_key=self.api_key,
            api_host="streaming.assemblyai.com"
        )
        
        client = StreamingClient(options)
        
        # Register event handlers
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Error, on_error)
        client.on(StreamingEvents.Termination, on_termination)
        
        return client
    
    def start_streaming(self):
        """Start continuous streaming transcription."""
        if self.is_streaming:
            print("Already streaming")
            return
        
        self.is_streaming = True
        self.is_paused = False
        
        # Create streaming client
        self.streaming_client = self._create_streaming_client()
        
        # Configure streaming parameters
        params = StreamingParameters(
            sample_rate=self.sample_rate,
            format_turns=True
        )
        
        # Connect to AssemblyAI
        self.streaming_client.connect(params)
        
        # Start streaming in a thread
        self.streaming_thread = threading.Thread(target=self._stream_audio)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
        print("Streaming started")
    
    def _stream_audio(self):
        """Stream audio to AssemblyAI."""
        try:
            print("Opening microphone stream...")
            
            # Create a generator that respects streaming state
            def audio_generator():
                # Calculate buffer size for ~100ms chunks (optimal for AssemblyAI)
                # At 16kHz, 100ms = 1600 samples
                buffer_size = 1600
                
                stream = self.pa.open(
                    rate=self.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=buffer_size
                )
                
                try:
                    while self.is_streaming:
                        # Read audio chunk (100ms at 16kHz)
                        audio_data = stream.read(buffer_size, exception_on_overflow=False)
                        
                        # Only yield if not paused
                        if not self.is_paused:
                            yield audio_data
                        else:
                            # Still read to prevent buffer overflow, but don't send
                            time.sleep(0.01)
                finally:
                    stream.stop_stream()
                    stream.close()
            
            # Stream the audio generator
            self.streaming_client.stream(audio_generator())
            
        except Exception as e:
            print(f"Error in audio streaming: {e}")
        finally:
            print("Microphone stream ended")
    
    def pause_streaming(self):
        """Pause streaming (audio still captured but not sent)."""
        if self.is_streaming:
            self.is_paused = True
            print("Streaming paused")
    
    def resume_streaming(self):
        """Resume streaming."""
        if self.is_streaming:
            self.is_paused = False
            print("Streaming resumed")
    
    def stop_streaming(self):
        """Stop streaming completely."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Disconnect streaming client
        if self.streaming_client:
            try:
                self.streaming_client.disconnect(terminate=True)
            except Exception as e:
                print(f"Error disconnecting: {e}")
            self.streaming_client = None
        
        # Wait for streaming thread to finish
        if self.streaming_thread:
            self.streaming_thread.join(timeout=2)
            self.streaming_thread = None
        
        print("Streaming stopped")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.stop_streaming()
        except Exception as e:
            print(f"Error stopping streaming during cleanup: {e}")
        
        try:
            if self.pa:
                self.pa.terminate()
                self.pa = None
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")


class SimpleAudioRecorder:
    """Simple audio recorder for fixed-duration recording with post-processing transcription."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize simple audio recorder.
        
        Args:
            sample_rate: Audio sample rate (16000 or higher recommended)
        """
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment")
        
        aai.settings.api_key = self.api_key
        self.sample_rate = sample_rate
        
        # Audio setup
        self.pa = pyaudio.PyAudio()
        
        # AssemblyAI transcriber
        self.transcriber = aai.Transcriber()
    
    def record_audio(self, duration: float = 5.0) -> bytes:
        """
        Record audio for a specified duration.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Raw audio data as bytes
        """
        stream = self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1024
        )
        
        print(f"Recording for {duration} seconds...")
        frames = []
        
        try:
            for _ in range(int(self.sample_rate / 1024 * duration)):
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()
        
        return b''.join(frames)
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio data using AssemblyAI.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Transcribed text
        """
        try:
            # Create config for transcription
            config = aai.TranscriptionConfig(
                sample_rate=self.sample_rate,
            )
            
            # Transcribe the audio
            transcript = self.transcriber.transcribe(
                audio_data,
                config=config
            )
            
            if transcript.status == aai.TranscriptStatus.error:
                print(f"Transcription error: {transcript.error}")
                return ""
            
            return transcript.text or ""
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def cleanup(self):
        """Clean up resources."""
        if self.pa:
            self.pa.terminate()