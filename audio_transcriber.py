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
                 on_transcript: Optional[Callable[[str], None]] = None,
                 on_final_transcript: Optional[Callable[[str], None]] = None,
                 sample_rate: int = 16000,
                 silence_threshold: int = 500):
        """
        Initialize real-time audio transcriber with AssemblyAI.
        
        Args:
            on_transcript: Callback for partial transcripts
            on_final_transcript: Callback for final transcripts
            sample_rate: Audio sample rate (16000 or higher recommended)
            silence_threshold: Silence duration in ms to end turn
        """
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment")
        
        aai.settings.api_key = self.api_key
        
        self.on_transcript = on_transcript
        self.on_final_transcript = on_final_transcript
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.transcript_buffer = []
        
        # Audio setup
        self.pa = pyaudio.PyAudio()
        self.audio_stream = None
        
        # Streaming client
        self.streaming_client = None
        self.session_id = None
        
    def _create_streaming_client(self):
        """Create AssemblyAI streaming client with event handlers."""
        
        def on_begin(client: StreamingClient, event: BeginEvent):
            self.session_id = event.id
            print(f"Transcription session started: {event.id}")
        
        def on_turn(client: StreamingClient, event: TurnEvent):
            """Handle turn events (complete utterances)."""
            transcript = event.transcript
            if transcript:
                self.transcript_buffer.append(transcript)
                
                if self.on_transcript:
                    self.on_transcript(transcript)
                
                # Final transcript when turn ends
                if hasattr(event, 'is_final') and event.is_final:
                    full_transcript = " ".join(self.transcript_buffer)
                    if self.on_final_transcript:
                        self.on_final_transcript(full_transcript)
                    self.transcript_buffer.clear()
        
        def on_error(client: StreamingClient, error: StreamingError):
            print(f"Transcription error: {error}")
        
        def on_termination(client: StreamingClient, event: TerminationEvent):
            print(f"Session terminated: {event.id}")
        
        # Create client with event handlers
        options = StreamingClientOptions(
            api_key=self.api_key
        )
        
        client = StreamingClient(options)
        
        # Register event handlers
        client.on(StreamingEvents.BEGIN, on_begin)
        client.on(StreamingEvents.TURN, on_turn)
        client.on(StreamingEvents.ERROR, on_error)
        client.on(StreamingEvents.TERMINATION, on_termination)
        
        return client
    
    def start_recording(self, duration: Optional[float] = None):
        """
        Start recording and transcribing audio.
        
        Args:
            duration: Optional duration in seconds (None for continuous)
        """
        if self.is_recording:
            print("Already recording")
            return
        
        self.is_recording = True
        self.transcript_buffer.clear()
        
        # Create streaming client
        self.streaming_client = self._create_streaming_client()
        
        # Configure streaming parameters
        params = StreamingSessionParameters(
            sample_rate=self.sample_rate,
            encoding="pcm_s16le",
            min_end_of_turn_silence_when_confident=self.silence_threshold,
            confidence_threshold=0.8,
            return_formatted_transcripts=True
        )
        
        # Connect to AssemblyAI
        self.streaming_client.connect(params)
        
        # Start audio stream
        self.audio_stream = self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )
        
        print("Recording and transcribing...")
        
        # Handle duration if specified
        if duration:
            threading.Timer(duration, self.stop_recording).start()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback to send audio to AssemblyAI."""
        if self.is_recording and self.streaming_client:
            try:
                # Send audio data to AssemblyAI
                self.streaming_client.send_audio(in_data)
            except Exception as e:
                print(f"Error sending audio: {e}")
        
        return (None, pyaudio.paContinue)
    
    def stop_recording(self):
        """Stop recording and get final transcript."""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        # Disconnect streaming client
        if self.streaming_client:
            self.streaming_client.disconnect()
            self.streaming_client = None
        
        print("Recording stopped")
        
        # Return accumulated transcript
        final_transcript = " ".join(self.transcript_buffer)
        self.transcript_buffer.clear()
        
        return final_transcript
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_recording()
        
        if self.pa:
            self.pa.terminate()


class SimpleAudioRecorder:
    """Simple audio recorder for fixed-duration recording with post-processing transcription."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize simple audio recorder.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.pa = pyaudio.PyAudio()
        self.audio_data = []
        self.is_recording = False
        
        # AssemblyAI setup
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if self.api_key:
            aai.settings.api_key = self.api_key
    
    def record_audio(self, duration: float = 5.0) -> bytes:
        """
        Record audio for specified duration.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio as bytes
        """
        print(f"Recording for {duration} seconds...")
        
        self.audio_data = []
        self.is_recording = True
        
        stream = self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1024
        )
        
        # Calculate number of chunks
        chunks_per_second = self.sample_rate / 1024
        total_chunks = int(chunks_per_second * duration)
        
        for _ in range(total_chunks):
            if not self.is_recording:
                break
            data = stream.read(1024, exception_on_overflow=False)
            self.audio_data.append(data)
        
        stream.stop_stream()
        stream.close()
        
        print("Recording complete")
        
        # Combine audio chunks
        audio_bytes = b''.join(self.audio_data)
        return audio_bytes
    
    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio using AssemblyAI (post-processing).
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Transcribed text
        """
        if not self.api_key:
            return "No API key configured for transcription"
        
        # Save audio to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import wave
            
            # Write WAV file
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_bytes)
            
            # Transcribe using AssemblyAI
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(tmp_file.name)
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            if transcript.status == aai.TranscriptStatus.error:
                return f"Transcription error: {transcript.error}"
            
            return transcript.text or "No speech detected"
    
    def cleanup(self):
        """Clean up resources."""
        self.is_recording = False
        if self.pa:
            self.pa.terminate()


if __name__ == "__main__":
    # Example 1: Real-time streaming transcription
    print("Example 1: Real-time streaming transcription")
    print("-" * 50)
    
    def on_partial_transcript(text):
        print(f"Partial: {text}")
    
    def on_final_transcript(text):
        print(f"Final: {text}")
    
    transcriber = AudioTranscriber(
        on_transcript=on_partial_transcript,
        on_final_transcript=on_final_transcript
    )
    
    try:
        transcriber.start_recording(duration=10)  # Record for 10 seconds
        time.sleep(11)  # Wait for recording to complete
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        transcriber.cleanup()
    
    print("\n" + "=" * 50 + "\n")
    
    # Example 2: Simple recording with post-transcription
    print("Example 2: Simple recording with post-transcription")
    print("-" * 50)
    
    recorder = SimpleAudioRecorder()
    
    try:
        # Record audio
        audio_data = recorder.record_audio(duration=5)
        
        # Transcribe
        print("Transcribing...")
        transcript = recorder.transcribe_audio(audio_data)
        print(f"Transcript: {transcript}")
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recorder.cleanup()