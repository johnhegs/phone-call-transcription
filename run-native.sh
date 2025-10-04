#!/bin/bash

# Phone Call Transcription - Native Setup & Run Helper
# This script automatically sets up and runs the transcription tool natively

set -e  # Exit on error (but we'll handle errors gracefully with traps)

# Color codes for output
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'
CYAN=$'\033[0;36m'
NC=$'\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}→ $1${NC}"
}

ask_yes_no() {
    local prompt="$1"
    local default="${2:-y}"

    if [ "$default" = "y" ]; then
        read -p "$(echo -e ${YELLOW}"$prompt (Y/n): "${NC})" choice
        choice=${choice:-y}
    else
        read -p "$(echo -e ${YELLOW}"$prompt (y/N): "${NC})" choice
        choice=${choice:-n}
    fi

    case "$choice" in
        y|Y ) return 0;;
        * ) return 1;;
    esac
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

OS_TYPE=$(detect_os)

# Welcome message
clear
print_header "Phone Call Transcription - Automated Native Setup"
echo "This script will automatically install and configure everything needed."
echo "You'll be asked for confirmation before any major installations."
echo ""
echo "Detected OS: $OS_TYPE"
echo ""

# Step 1: Check and install Python
print_header "Step 1: Setting Up Python 3.11+"

REQUIRED_PYTHON_VERSION="3.11"
PYTHON_CMD=""

# Check for python3 command
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        PYTHON_CMD="python3"
        print_success "Found Python $PYTHON_VERSION"
    else
        print_warning "Found Python $PYTHON_VERSION (need 3.11+)"

        # Try to install via pyenv
        if command -v pyenv &> /dev/null; then
            print_info "Using pyenv to install Python 3.11.13..."
            pyenv install -s 3.11.13
            pyenv local 3.11.13
            PYTHON_CMD="python3"
            print_success "Python 3.11.13 installed via pyenv"
        else
            print_warning "pyenv not found. Installing pyenv..."
            if ask_yes_no "Install pyenv to manage Python versions?"; then
                curl https://pyenv.run | bash

                # Add pyenv to current session
                export PYENV_ROOT="$HOME/.pyenv"
                export PATH="$PYENV_ROOT/bin:$PATH"
                eval "$(pyenv init -)"

                print_success "pyenv installed"
                print_info "Installing Python 3.11.13..."
                pyenv install 3.11.13
                pyenv local 3.11.13
                PYTHON_CMD="python3"
                print_success "Python 3.11.13 installed"
            else
                print_error "Cannot proceed without Python 3.11+. Please install manually."
                exit 1
            fi
        fi
    fi
else
    print_warning "Python 3 not found. Installing..."

    # Try to install pyenv first
    if ! command -v pyenv &> /dev/null; then
        if ask_yes_no "Install pyenv and Python 3.11.13?"; then
            print_info "Installing pyenv..."
            curl https://pyenv.run | bash

            # Add pyenv to current session
            export PYENV_ROOT="$HOME/.pyenv"
            export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init -)"

            print_success "pyenv installed"
        else
            print_error "Cannot proceed without Python. Please install manually."
            exit 1
        fi
    fi

    print_info "Installing Python 3.11.13..."
    pyenv install 3.11.13
    pyenv local 3.11.13
    PYTHON_CMD="python3"
    print_success "Python 3.11.13 installed"
fi

# Ensure pyenv is initialized if it exists
if command -v pyenv &> /dev/null; then
    eval "$(pyenv init -)"
    if [ -f ".python-version" ]; then
        EXPECTED_VERSION=$(cat .python-version)
        print_info "Using Python version from .python-version: $EXPECTED_VERSION"
        if ! pyenv versions | grep -q "$EXPECTED_VERSION"; then
            print_info "Installing required Python version $EXPECTED_VERSION..."
            pyenv install -s "$EXPECTED_VERSION"
        fi
        pyenv local "$EXPECTED_VERSION"
    fi
fi

echo ""

# Step 2: Setup virtual environment
print_header "Step 2: Setting Up Python Virtual Environment"

if [ -d "venv" ]; then
    print_success "Virtual environment already exists"
else
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

print_success "Virtual environment activated"
echo ""

# Step 3: Install Python dependencies
print_header "Step 3: Installing Python Dependencies"

print_info "Installing packages from requirements.txt..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

print_success "All Python dependencies installed"
echo ""

# Step 4: Install and setup Ollama
print_header "Step 4: Setting Up Ollama (AI Summarization)"

if command -v ollama &> /dev/null; then
    print_success "Ollama is already installed"
else
    print_warning "Ollama not found"

    if ask_yes_no "Install Ollama now? (~50MB download)"; then
        print_info "Installing Ollama..."

        if [ "$OS_TYPE" = "macos" ] || [ "$OS_TYPE" = "linux" ]; then
            curl -fsSL https://ollama.ai/install.sh | sh
            print_success "Ollama installed"
        else
            print_error "Automatic installation not supported on this OS"
            echo "Please visit https://ollama.ai to install manually"
            exit 1
        fi
    else
        print_warning "Skipping Ollama installation. AI summarization won't work."
        echo ""
        read -p "Press Enter to continue anyway..."
    fi
fi

# Start Ollama service if not running
if command -v ollama &> /dev/null; then
    print_info "Checking Ollama service..."

    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        print_success "Ollama service is running"
    else
        print_info "Starting Ollama service..."

        if [ "$OS_TYPE" = "macos" ]; then
            # On macOS, start in background
            nohup ollama serve > /dev/null 2>&1 &
            sleep 3
        elif [ "$OS_TYPE" = "linux" ]; then
            # On Linux, try systemd first, then background
            if systemctl is-active --quiet ollama 2>/dev/null; then
                sudo systemctl start ollama
            else
                nohup ollama serve > /dev/null 2>&1 &
                sleep 3
            fi
        fi

        # Verify it started
        sleep 2
        if curl -s http://localhost:11434/api/tags &> /dev/null; then
            print_success "Ollama service started"
        else
            print_warning "Could not auto-start Ollama. You may need to run 'ollama serve' manually."
        fi
    fi

    # Download llama3.1 model
    print_info "Checking for llama3.1 model..."

    if ollama list 2>/dev/null | grep -q "llama3.1"; then
        print_success "llama3.1 model is ready"
    else
        print_warning "llama3.1 model not found"

        if ask_yes_no "Download llama3.1 model? (~4GB, one-time download)"; then
            print_info "Downloading llama3.1 model (this may take several minutes)..."
            ollama pull llama3.1:latest
            print_success "Model downloaded successfully"
        else
            print_warning "Skipping model download. AI summarization won't work without it."
        fi
    fi
fi

echo ""

# Step 5: Create required directories
print_header "Step 5: Creating Required Directories"

for dir in calls_to_process calls_transcribed calls_summary calls_analysis; do
    if [ -d "$dir" ]; then
        print_success "$dir/ exists"
    else
        mkdir -p "$dir"
        print_success "$dir/ created"
    fi
done

echo ""

# Step 6: Check for audio files
print_header "Step 6: Ready to Process Audio Files"

AUDIO_FILES=$(find calls_to_process -maxdepth 1 -name "*.mp3" 2>/dev/null | wc -l | tr -d ' ')

if [ "$AUDIO_FILES" -eq 0 ]; then
    print_warning "No audio files found in calls_to_process/"
    echo ""
    echo "Please add your stereo MP3 files to the calls_to_process/ directory."
    echo "Files should have one speaker on the left channel and one on the right."
    echo ""

    if ask_yes_no "Open calls_to_process/ folder now?" "n"; then
        if [ "$OS_TYPE" = "macos" ]; then
            open calls_to_process/
        elif [ "$OS_TYPE" = "linux" ]; then
            xdg-open calls_to_process/ 2>/dev/null || echo "Please manually open: $(pwd)/calls_to_process/"
        fi
    fi

    echo ""
    echo "After adding files, you can run:"
    echo "  ${GREEN}./run-native.sh${NC}  (to run this setup script again)"
    echo ""
else
    print_success "Found $AUDIO_FILES audio file(s) ready to process"
    echo ""

    # List the files
    echo "Files found:"
    find calls_to_process -maxdepth 1 -name "*.mp3" -exec basename {} \; | sed 's/^/  • /'
    echo ""
fi

# Step 7: Run the application
print_header "Step 7: All Set! System Ready"

print_success "✓ Python environment configured"
print_success "✓ Dependencies installed"
print_success "✓ Ollama configured"
print_success "✓ Directories created"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$AUDIO_FILES" -gt 0 ]; then
    echo "What would you like to do?"
    echo ""
    echo "  ${CYAN}1${NC}) Process a single file"
    echo "  ${CYAN}2${NC}) Process all files in batch"
    echo "  ${CYAN}3${NC}) Test Ollama connection"
    echo "  ${CYAN}4${NC}) Configure settings"
    echo "  ${CYAN}5${NC}) Exit (I'll run commands manually)"
    echo ""

    read -p "$(echo -e ${YELLOW}"Enter your choice (1-5): "${NC})" choice

    case "$choice" in
        1)
            echo ""
            echo "Available files:"
            find calls_to_process -maxdepth 1 -name "*.mp3" -exec basename {} \; | nl
            echo ""
            read -p "$(echo -e ${BLUE}"Enter the filename: "${NC})" filename

            # Strip path if user included it
            filename=$(basename "$filename")

            if [ -f "calls_to_process/$filename" ]; then
                echo ""
                print_info "Processing $filename..."
                echo ""
                python transcribe_and_summarise.py --file "$filename"

                echo ""
                print_success "Processing complete!"
                echo ""
                echo "Output files:"
                echo "  • Transcript: calls_transcribed/${filename%.mp3}.txt"
                echo "  • Summary: calls_summary/${filename%.mp3}.txt"
                echo "  • Analysis: calls_analysis/${filename%.mp3}.txt"
            else
                print_error "File not found: calls_to_process/$filename"
            fi
            ;;
        2)
            echo ""
            print_info "Processing all files in batch mode..."
            echo ""
            python transcribe_and_summarise.py --batch

            echo ""
            print_success "Batch processing complete!"
            echo ""
            echo "Check the output directories:"
            echo "  • calls_transcribed/"
            echo "  • calls_summary/"
            echo "  • calls_analysis/"
            ;;
        3)
            echo ""
            print_info "Testing Ollama connection..."
            echo ""
            python test_ollama.py
            ;;
        4)
            echo ""
            print_info "Opening configuration menu..."
            echo ""
            python configure.py
            ;;
        5|*)
            echo ""
            print_info "Setup complete! You can now run commands manually."
            ;;
    esac
else
    echo "Next steps:"
    echo ""
    echo "  ${CYAN}1.${NC} Add MP3 files to: ${GREEN}calls_to_process/${NC}"
    echo "  ${CYAN}2.${NC} Run this script again: ${GREEN}./run-native.sh${NC}"
    echo ""
    echo "Or activate the environment and run manually:"
    echo "  ${GREEN}source venv/bin/activate${NC}"
    echo "  ${GREEN}python transcribe_and_summarise.py --file your_file.mp3${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_success "Setup script finished!"
echo ""

# Show useful commands
echo "Useful commands for future use:"
echo ""
echo "  ${GREEN}source venv/bin/activate${NC}              - Activate Python environment"
echo "  ${GREEN}python transcribe_and_summarise.py --file <file>${NC}   - Process single file"
echo "  ${GREEN}python transcribe_and_summarise.py --batch${NC}         - Process all files"
echo "  ${GREEN}python test_ollama.py${NC}                 - Test Ollama connection"
echo "  ${GREEN}python configure.py${NC}                   - Configure settings"
echo "  ${GREEN}./run-native.sh${NC}                       - Run this setup script again"
echo ""
