#!/usr/bin/env python3
"""
Test script for the refactored voice assistant architecture.
"""

import time
import queue
from audio_transcriber import AudioTranscriber

def test_streaming_with_pause():
    """Test streaming with pause/resume functionality."""
    print("Testing streaming with pause/resume...")
    
    # Create a queue to receive transcripts
    transcript_queue = queue.Queue()
    
    # Initialize transcriber
    transcriber = AudioTranscriber(transcript_queue=transcript_queue)
    
    try:
        # Start streaming
        print("\n1. Starting streaming...")
        transcriber.start_streaming()
        
        # Stream for 5 seconds
        print("2. Streaming for 5 seconds (speak now)...")
        time.sleep(5)
        
        # Pause streaming
        print("\n3. Pausing streaming...")
        transcriber.pause_streaming()
        print("   Paused for 3 seconds (audio not sent to API)...")
        time.sleep(3)
        
        # Resume streaming
        print("\n4. Resuming streaming...")
        transcriber.resume_streaming()
        print("   Streaming for another 5 seconds (speak now)...")
        time.sleep(5)
        
        # Stop streaming
        print("\n5. Stopping streaming...")
        transcriber.stop_streaming()
        
        # Check for transcripts
        print("\n6. Transcripts received:")
        transcripts = []
        while not transcript_queue.empty():
            try:
                transcript = transcript_queue.get_nowait()
                transcripts.append(transcript)
                print(f"   - {transcript}")
            except queue.Empty:
                break
        
        if not transcripts:
            print("   No transcripts received (no speech detected)")
        
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
    
    finally:
        transcriber.cleanup()

def test_queue_communication():
    """Test queue-based communication between threads."""
    print("Testing queue-based communication...")
    
    # Create a shared queue
    transcript_queue = queue.Queue()
    
    # Initialize transcriber with queue
    transcriber = AudioTranscriber(transcript_queue=transcript_queue)
    
    try:
        print("\n1. Starting streaming...")
        transcriber.start_streaming()
        
        print("2. Speak something and wait for end of turn...")
        print("   (Say a complete sentence then pause)")
        
        # Wait for a transcript with timeout
        try:
            transcript = transcript_queue.get(timeout=10)
            print(f"\n✓ Received transcript via queue: '{transcript}'")
        except queue.Empty:
            print("\n- No transcript received (timeout)")
        
        transcriber.stop_streaming()
        print("\n✓ Queue communication test completed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
    
    finally:
        transcriber.cleanup()

if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("Voice Assistant Architecture Test")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "queue":
        test_queue_communication()
    else:
        test_streaming_with_pause()
    
    print("\nAll tests completed!")