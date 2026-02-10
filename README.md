# MedScribe AI

Turn medical audio recordings into structured SOAP notes. Runs locally, keeps your data private.

## What it does

1. **Transcribes** audio (Whisper)
2. **Identifies speakers** - Doctor vs Patient (optional)
3. **Generates SOAP notes** (Ollama LLM)

Medical conversations in, clinical documentation out.

## Quick Start

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- [FFmpeg](https://ffmpeg.org/download.html) (`brew install ffmpeg` on Mac)

### Setup

```bash
# Clone and install
git clone <your-repo-url>
cd files
pip install -r requirements.txt

# Get an Ollama model
ollama pull qwen3:14b

# Configure
cp .env.example .env
# Edit .env - set your Ollama model name
```

### Run It

**CLI (simple file processing):**
```bash
python -m cli audio.mp3
python -m cli audio.mp3 --output ./my_notes
python -m cli --text "Patient presents with..."
```

**API (for web interface & async jobs):**
```bash
# Make sure Ollama is running first
ollama serve

# Then in another terminal:
docker-compose up
# Open http://localhost:8000/api/docs
# That's it - Redis, API, and Celery all start together
```

## Configuration

Copy `.env.example` to `.env` and configure:

**Required:**
- `MedScribe_WHISPER_MODEL` - Transcription model (tiny/base/small/medium/large)
- `MedScribe_OLLAMA_MODEL` - Your Ollama model name (e.g., qwen3:14b, llama3.2)
- `MedScribe_OLLAMA_BASE_URL` - Ollama server URL
  - Local: `http://localhost:11434`
  - Docker: `http://host.docker.internal:11434`

**Optional:**
- `MedScribe_HUGGINGFACE_TOKEN` - Enables speaker diarization (Doctor vs Patient labels)
  - Get token at: https://huggingface.co/settings/tokens
  - Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1

**Note:** Redis and Celery are auto-configured by docker-compose. All other settings have sensible defaults. See [config.py](config.py) for advanced options.

## API Features

- Async job processing (upload audio, get results later)
- WebSocket for real-time progress
- Health checks

## Troubleshooting

**"Ollama connection failed"** - Make sure Ollama is running (`ollama serve`). For docker-compose, use `http://host.docker.internal:11434` in `.env`

**"Connection refused"** - Check Ollama URL in `.env` (`localhost` vs `host.docker.internal`)

**Poor transcription** - Use bigger Whisper model: `MedScribe_WHISPER_MODEL=medium`

**Missing speaker labels** - Set `MedScribe_HUGGINGFACE_TOKEN` in `.env` (optional feature)

## License

MIT

## Built With

- [Whisper](https://github.com/openai/whisper) - Transcription
- [Pyannote](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Ollama](https://ollama.ai/) - Local LLM
- [FastAPI](https://fastapi.tiangolo.com/) - API
- [LangChain](https://www.langchain.com/) - LLM orchestration

---

**Note:** For learning/research only. Not a medical device. Don't use for actual clinical work without proper validation.
