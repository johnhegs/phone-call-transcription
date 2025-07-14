# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open source preparation and documentation
- Community guidelines and contribution standards
- Security and privacy considerations

### Changed
- Converted all text to UK English spelling
- Enhanced README with comprehensive documentation
- Improved Docker integration and commands

### Removed
- Domain-specific references and example content
- Development-only demonstration files

## [1.0.0] - 2025-07-14

### Added
- **Core Features**
  - Stereo audio channel separation for different speakers
  - OpenAI Whisper integration for speech-to-text conversion
  - Chronological transcript reconstruction with precise timestamps
  - Ollama LLM integration for AI-powered call summarisation
  - Conversation analysis with speaking time and turn statistics
  
- **Docker Support**
  - Fully containerised deployment with Docker Compose
  - Isolated Ollama service with persistent model storage
  - Pre-cached Whisper models for instant loading
  - Convenient Docker command scripts
  - Health checks and service coordination

- **Batch Processing**
  - Process multiple audio files automatically
  - Skip existing files or force reprocessing
  - Progress tracking and failure handling
  - Configurable input/output directories

- **Configuration System**
  - Flexible configuration via config files
  - Customisable speaker labels and model settings
  - Interactive configuration interface
  - Multiple prompt templates for different use cases

- **Output Formats**
  - Professional transcript formatting with timestamps
  - AI-generated summaries with key insights
  - Conversation analysis with speaking statistics
  - Organised output directory structure

### Technical Details
- **Python 3.11+** with Poetry dependency management
- **OpenAI Whisper** for high-quality speech recognition
- **Ollama** for local LLM processing
- **Docker** containerisation for easy deployment
- **Stereo MP3** support with left/right channel separation

### Documentation
- Comprehensive README with installation and usage guides
- Docker deployment instructions
- Configuration examples and templates
- Troubleshooting and FAQ sections

---

## Release Notes Format

### Types of Changes
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

### Version Numbering
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes
