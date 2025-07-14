#!/bin/bash

# Docker run script for Transcribe application
# This script provides convenient commands for running the Docker setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to setup directories
setup_directories() {
    print_status "Setting up directories..."
    mkdir -p calls_to_process calls_transcribed calls_summary calls_analysis
    
    # Create default config.txt if it doesn't exist
    if [ ! -f "config.txt" ]; then
        print_status "Creating default config.txt..."
        cat > config.txt << 'EOF'
# AI Model Settings
OLLAMA_MODEL=llama3.1:latest
WHISPER_MODEL=base

# Speaker Labels
LEFT_SPEAKER_NAME=Customer
RIGHT_SPEAKER_NAME=Agent

# Processing Settings
SEGMENT_MERGE_THRESHOLD=2.0

# Language Settings
FORCE_LANGUAGE=auto
EOF
    fi
    
    # Create default prompt template if it doesn't exist
    if [ ! -f "prompt_template.txt" ]; then
        print_status "Creating default prompt_template.txt..."
        cat > prompt_template.txt << 'EOF'
Please analyze the following phone call transcript and provide:

1. **Key Topics Discussed**: Main subjects covered in the call
2. **Action Items**: Any tasks, commitments, or follow-ups mentioned
3. **Decisions Made**: Any conclusions or decisions reached
4. **Next Steps**: What should happen next based on the conversation

Call Transcript:
{TRANSCRIPT}

Please provide a detailed analysis:
EOF
    fi
}

# Function to start services
start_services() {
    print_status "Starting Ollama service..."
    docker-compose up -d ollama
    
    print_status "Waiting for Ollama to be ready..."
    timeout=300
    counter=0
    while ! docker-compose exec ollama ollama list > /dev/null 2>&1; do
        if [ $counter -ge $timeout ]; then
            print_error "Ollama service failed to start within $timeout seconds"
            exit 1
        fi
        sleep 5
        counter=$((counter + 5))
        echo -n "."
    done
    echo
    
    print_status "Pulling Ollama model (this may take a few minutes)..."
    docker-compose exec ollama ollama pull llama3.1:latest
    
    print_status "Services are ready!"
}

# Function to run single file processing
run_single() {
    local file="$1"
    if [ -z "$file" ]; then
        print_error "Please specify a file to process"
        echo "Usage: $0 single <filename>"
        exit 1
    fi
    
    if [ ! -f "calls_to_process/$file" ]; then
        print_error "File calls_to_process/$file not found"
        exit 1
    fi
    
    print_status "Processing single file: $file"
    docker-compose run --rm transcribe python transcribe_and_summarise.py --file "$file"
}

# Function to run batch processing
run_batch() {
    print_status "Running batch processing..."
    docker-compose --profile batch up transcribe-batch
}

# Function to run interactive shell
run_shell() {
    print_status "Starting interactive shell..."
    docker-compose run --rm transcribe bash
}

# Function to show logs
show_logs() {
    docker-compose logs -f "$1"
}

# Function to stop services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down
}

# Function to clean up everything
cleanup() {
    print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_status "Cleanup complete"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main script logic
check_docker

case "${1:-help}" in
    "setup")
        setup_directories
        ;;
    "start")
        setup_directories
        start_services
        ;;
    "single")
        start_services
        run_single "$2"
        ;;
    "batch")
        start_services
        run_batch
        ;;
    "shell")
        start_services
        run_shell
        ;;
    "logs")
        show_logs "$2"
        ;;
    "stop")
        stop_services
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        echo "Usage: $0 {setup|start|single|batch|shell|logs|stop|cleanup|help}"
        echo ""
        echo "Commands:"
        echo "  setup          - Create necessary directories and default config files"
        echo "  start          - Start all services (Ollama + dependencies)"
        echo "  single <file>  - Process a single audio file"
        echo "  batch          - Process all files in calls_to_process directory"
        echo "  shell          - Start interactive shell in transcribe container"
        echo "  logs [service] - Show logs for all services or specific service"
        echo "  stop           - Stop all services"
        echo "  cleanup        - Remove all containers, images, and volumes"
        echo "  help           - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 setup                    # Initial setup"
        echo "  $0 start                    # Start services"
        echo "  $0 single recording.mp3     # Process single file"
        echo "  $0 batch                    # Process all files"
        echo "  $0 logs ollama              # Show Ollama logs"
        ;;
esac