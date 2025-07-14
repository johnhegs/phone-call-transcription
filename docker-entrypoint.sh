#!/bin/bash
set -e

# Wait for Ollama to be ready if OLLAMA_URL is set
if [ -n "$OLLAMA_URL" ]; then
    echo "Waiting for Ollama service to be ready..."
    until curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; do
        echo "Ollama not ready, waiting 5 seconds..."
        sleep 5
    done
    echo "Ollama service is ready!"
    
    # Check if the model is available, pull if not
    MODEL_NAME=${OLLAMA_MODEL:-llama3.1:latest}
    echo "Checking if model $MODEL_NAME is available..."
    
    # Check if model exists using API call
    if ! curl -sf "$OLLAMA_URL/api/tags" | grep -q "$MODEL_NAME"; then
        echo "Model $MODEL_NAME not found, but will be handled by the main process..."
    else
        echo "Model $MODEL_NAME is already available"
    fi
fi

# Execute the main command
exec "$@"