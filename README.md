# C.R.A.I.G (Conversational Room Assistant & Intelligence Gateway)

A Python-based voice assistant that uses wake word detection, real-time speech transcription (AssemblyAI), LLM processing (OpenAI/Anthropic), and text-to-speech output.

## Features

- **Wake Word Detection**: Supports both Picovoice Porcupine and OpenWakeWord
- **Real-time Transcription**: Uses AssemblyAI's streaming API for low-latency transcription
- **Multiple LLM Providers**: OpenAI GPT and Anthropic Claude support
- **Text-to-Speech**: Multiple TTS engines (pyttsx3, gTTS, system TTS)
- **Cross-platform**: Works on macOS, Linux, and Raspberry Pi

## Architecture

```
Wake Word Detection → Audio Recording → Speech Transcription → LLM Processing → TTS Output
```

## Prerequisites

### macOS
```bash
brew install portaudio
```

### Linux/Raspberry Pi
```bash
sudo apt-get install portaudio19-dev
sudo apt-get install espeak  # Optional: for system TTS
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo>
cd max
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- `ASSEMBLYAI_API_KEY`: For speech transcription (get from https://www.assemblyai.com)
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: For LLM responses
- `PICOVOICE_ACCESS_KEY`: Optional, for Porcupine wake word (get from https://picovoice.ai)

## Usage

### Basic Usage
```bash
python voice_assistant.py
```

### Command Line Options
```bash
# Use different wake word
python voice_assistant.py --wake-word alexa

# Use OpenWakeWord instead of Porcupine
python voice_assistant.py --no-porcupine

# Use Anthropic Claude instead of OpenAI
python voice_assistant.py --llm anthropic --model claude-3-5-haiku-20241022

# Use system TTS
python voice_assistant.py --tts system

# Adjust recording duration
python voice_assistant.py --duration 10

# Disable streaming transcription
python voice_assistant.py --no-streaming
```

### Available Wake Words (Porcupine)
- jarvis
- alexa
- hey_siri
- ok_google
- Custom .ppn files

### Available Wake Words (OpenWakeWord)
- alexa
- hey_mycroft
- Custom models

## Module Descriptions

### `wake_word_detector.py`
Implements wake word detection using either Picovoice Porcupine (commercial, high accuracy) or OpenWakeWord (open-source, no API key required).

### `audio_transcriber.py`
Handles audio recording and transcription using AssemblyAI's streaming API. Supports both real-time streaming and post-processing modes.

### `llm_providers.py`
Manages LLM interactions with support for OpenAI GPT and Anthropic Claude. Maintains conversation history and supports streaming responses.

### `tts_engine.py`
Provides text-to-speech functionality with multiple engine options:
- pyttsx3: Offline, cross-platform
- gTTS: Google TTS, requires internet
- System: Uses OS-native TTS commands

### `voice_assistant.py`
Main integration that combines all modules into a complete voice assistant.

## Raspberry Pi Considerations

1. **Performance**: Use lighter models (gpt-4o-mini, claude-3-5-haiku) for better performance
2. **Audio Setup**: Ensure your microphone is properly configured:
   ```bash
   # Test microphone
   arecord -l  # List recording devices
   arecord -D plughw:1,0 -d 5 test.wav  # Test recording
   ```
3. **Wake Word**: OpenWakeWord may perform better than Porcupine on lower-end Pi models
4. **TTS**: Use pyttsx3 or espeak for better performance

## Troubleshooting

### No Audio Input
- Check microphone permissions
- Verify PyAudio installation: `python -c "import pyaudio"`
- Test with system tools: `arecord` (Linux) or Audio MIDI Setup (macOS)

### API Key Errors
- Ensure all required keys are in `.env` file
- Check API key validity and quota limits

### Wake Word Not Detected
- Adjust sensitivity in code (0.0-1.0)
- Try different wake words
- Check microphone quality and positioning

### TTS Not Working
- Install system TTS: `apt-get install espeak` (Linux)
- Try different engines: `--tts system` or `--tts gtts`

## Performance Optimization

1. **Reduce Latency**:
   - Use streaming transcription
   - Choose faster LLM models
   - Use local TTS (pyttsx3)

2. **Reduce Resource Usage**:
   - Lower audio sample rate (minimum 16kHz)
   - Use shorter recording durations
   - Disable streaming for lower CPU usage

3. **Improve Accuracy**:
   - Use high-quality microphone
   - Reduce background noise
   - Adjust wake word sensitivity

## Future Enhancements

- Image-based wake detection (as mentioned)
- Multi-language support
- Custom voice commands
- Integration with smart home devices
- Local LLM support (Ollama)
- Voice activity detection (VAD)
- Speaker diarization
- Conversation memory persistence

## License

MIT License - See LICENSE file for details