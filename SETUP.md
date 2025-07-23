# Setup Instructions

This project uses standard Python pip with virtual environments for dependency management.

## Quick Setup

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Test Installation
```bash
python test_faster_whisper.py
```

### 4. Install Ollama (for AI summaries)
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.1
```

## Development Setup

For development work, install additional tools:

```bash
pip install -r requirements-dev.txt
```

This includes:
- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Code linting

## Usage

Always activate the virtual environment before running the transcription script:

```bash
source venv/bin/activate
python transcribe_and_summarise.py --file your_audio.mp3
```

## Key Features

- ✅ Uses `faster-whisper` for high-performance transcription on all platforms
- ✅ Supports x86_64, ARM64, Windows, Linux, and macOS
- ✅ Simple dependency management with standard pip

## Dependencies

### Production (`requirements.txt`)
- `faster-whisper==1.1.1` - High-performance transcription engine
- `pydub==0.25.1` - Audio processing
- `SpeechRecognition==3.10.0` - Fallback transcription
- `requests==2.31.0` - HTTP requests for Ollama API
- `tqdm==4.67.0` - Progress bars

### Development (`requirements-dev.txt`)
- `pytest>=7.4.0` - Testing framework
- `black>=23.0.0` - Code formatter
- `flake8>=6.0.0` - Code linter
- Plus all production dependencies via `-r requirements.txt`
