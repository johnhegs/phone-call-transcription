FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Configure poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only=main

# Install openai-whisper separately (as mentioned in README)
RUN pip install openai-whisper

# Pre-download the base Whisper model to optimize runtime performance
# Medium model will be downloaded on first use to avoid Docker build memory issues
RUN python -c "import whisper; print('Downloading Whisper base model...'); whisper.load_model('base'); print('Base model cached successfully')"

# Copy application files
COPY transcribe_and_summarise.py configure.py test_ollama.py ./
COPY configure.py .
COPY test_ollama.py .

# Create necessary directories
RUN mkdir -p calls_to_process calls_transcribed calls_summary calls_analysis

# Copy entrypoint script
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Set default command
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "transcribe_and_summarise.py", "--batch"]
