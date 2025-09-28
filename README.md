# Craig - Voice Assistant

A witty, sarcastic voice assistant with custom wake word detection, real-time speech transcription (AssemblyAI), LLM processing (OpenAI/Anthropic/Groq), and high-quality text-to-speech (ElevenLabs). Features conversation mode with idle timeout and persistent memory.

## Features

- **Custom Wake Word Detection**: Trained TensorFlow Lite model specifically for "Craig" wake word
- **Conversation Mode**: Seamless conversation flow with configurable idle timeout
- **Real-time Transcription**: AssemblyAI's streaming API for low-latency speech recognition
- **Multiple LLM Providers**: OpenAI GPT, Anthropic Claude, and Groq support with conversation history
- **High-Quality TTS**: ElevenLabs voices with natural speech synthesis
- **Personality**: Witty, sarcastic personality with persistent conversation memory
- **Cross-platform**: Works on macOS, Linux, and Raspberry Pi
- **Tooling**: Built-in weather lookup, Twilio texting, and rainbow text console effects for richer interactions

## Architecture

Craig operates in two main modes with seamless transitions:

```
┌─────────────────┐     Wake Word     ┌──────────────────┐
│   Wake Word     │ ────────────────► │  Conversation    │
│   Detection     │                   │     Mode         │
│   (Sleeping)    │ ◄───────────────  │   (Active)       │
└─────────────────┘   Idle Timeout    └──────────────────┘
         ▲                                         │
         │                                         │
         └─────────────────────────────────────────┘
                 Persistent Conversation History
```

**Wake Word Mode**: Listens for "Craig" wake word using custom TensorFlow Lite model
**Conversation Mode**: Real-time speech transcription → LLM processing → ElevenLabs TTS

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
cd craig
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
- `ELEVEN_LABS_KEY`: For high-quality text-to-speech (get from https://elevenlabs.io)
- `OPENWEATHER_API_KEY`: For the weather lookup tool (get a free key at https://openweathermap.org/api)
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`, `MY_PHONE_NUMBER`: For the SMS tool (generate in the Twilio Console)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GROQ_API_KEY`: For LLM responses
  - OpenAI: https://platform.openai.com/api-keys
  - Anthropic: https://console.anthropic.com/
  - Groq: https://console.groq.com/keys

## Usage

### Basic Usage

```bash
python assistant.py
```

### Command Line Options

```bash
# Use different LLM provider (default: groq)
python assistant.py --llm openai --model gpt-4o

# Use Anthropic Claude
python assistant.py --llm anthropic --model claude-3-5-haiku-20241022

# Use different ElevenLabs voice
python assistant.py --voice L0Dsvb3SLTyegXwtm47J

# Adjust idle timeout before returning to wake word mode
python assistant.py --timeout 15
```

### Wake Word

Craig uses a custom-trained TensorFlow Lite model specifically for the wake word "Craig". The model files are located in the `models/` directory:

- `craig.tflite`: The trained model
- `craig_scaler.pkl`: Feature scaler used during training

### Conversation Commands

While in conversation mode, you can use these commands:

- "goodbye", "bye", "exit", "stop", "go to sleep" - Exit conversation mode
- "clear history", "forget everything", "reset memory", "wipe memory" - Clear conversation history
- "how many conversations", "conversation count", "history length" - Check conversation history length

## Module Descriptions

### `assistant.py`

Main voice assistant implementation featuring conversation mode with idle timeout. Manages state transitions between wake word detection and conversation modes, maintains persistent conversation history, and coordinates all components.

### `wake_word_detector.py`

Custom wake word detector using TensorFlow Lite models. Trained specifically for "Craig" wake word detection with real-time audio processing and configurable sensitivity thresholds.

### `audio_transcriber.py`

Handles real-time audio recording and speech transcription using AssemblyAI's streaming API. Features queue-based communication for seamless integration with conversation flow and supports pause/resume functionality.

### `llm_providers.py`

LLM management system supporting OpenAI GPT, Anthropic Claude, and Groq providers. Maintains persistent conversation history with pickle-based storage, supports streaming responses, and includes abstract provider architecture.

### `tts_engine.py`

ElevenLabs-powered text-to-speech engine with high-quality voice synthesis. Features asynchronous speech generation, pygame-based audio playback, and support for multiple ElevenLabs voice models.

## Raspberry Pi Considerations

1. **Performance**: Use Groq or lighter models (claude-3-5-haiku, gpt-4o-mini) for best performance on Pi hardware
2. **Hardware**: Raspberry Pi 4 or 5 recommended for TensorFlow Lite inference
3. **Audio Setup**: Ensure your microphone is properly configured:
   ```bash
   # Test microphone
   arecord -l  # List recording devices
   arecord -D plughw:1,0 -d 5 test.wav  # Test recording
   ```
4. **Wake Word**: Custom TensorFlow Lite model is optimized but may require Pi 4+ for real-time performance
5. **TTS**: ElevenLabs works well on Pi with good internet connection; responses are processed server-side

## Troubleshooting

### No Audio Input

- Check microphone permissions
- Verify PyAudio installation: `python -c "import pyaudio"`
- Test with system tools: `arecord` (Linux) or Audio MIDI Setup (macOS)

### API Key Errors

- Ensure all required keys are in `.env` file
- Check API key validity and quota limits

### Wake Word Not Detected

- Ensure model files exist: `models/craig.tflite` and `models/craig_scaler.pkl`
- Adjust threshold in `wake_word_detector.py` (default: 0.520)
- Retrain model if accuracy is poor (see training notebook)
- Check microphone quality and positioning
- Ensure TensorFlow Lite is properly installed

### ElevenLabs TTS Not Working

- Check `ELEVEN_LABS_KEY` in `.env` file
- Verify internet connection (ElevenLabs requires online access)
- Ensure pygame is installed: `pip install pygame`
- Try different voice IDs from ElevenLabs dashboard

## Performance Optimization

1. **Reduce Latency**:

   - Use Groq for fastest LLM responses (default)
   - Use lighter models: `claude-3-5-haiku` or `gpt-4o-mini`
   - Keep conversation responses concise (Craig's personality helps with this)
   - Adjust idle timeout based on your usage patterns

2. **Reduce Resource Usage**:

   - TensorFlow Lite model is optimized for edge devices
   - AssemblyAI streaming uses minimal CPU when idle
   - ElevenLabs TTS requires internet but processes quickly
   - Use appropriate hardware (Raspberry Pi 4+ recommended)

3. **Improve Accuracy**:
   - Use high-quality microphone with good noise isolation
   - Position microphone appropriately for wake word detection
   - Retrain wake word model if false positives/negatives occur
   - Ensure consistent audio levels and background noise

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
