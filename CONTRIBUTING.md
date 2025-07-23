# Contributing to Phone Call Transcription & Summarisation

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### 🐛 Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, please include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected vs actual behaviour**
- **Environment details** (OS, Python version, Docker version)
- **Relevant logs** or error messages
- **Audio file characteristics** (if applicable)

### 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear title** and detailed description
- **Use case** or problem it solves
- **Proposed solution** (if you have one)
- **Alternative solutions** considered

### 🔧 Code Contributions

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request**

#### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/phone-call-transcriber.git
cd phone-call-transcriber

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install faster-whisper

# Install development dependencies (optional)
pip install pytest black flake8

# Run tests
pytest

# Run linting
flake8 .
black .
```

#### Coding Standards

- **UK English spelling** throughout (analyse, summarise, etc.)
- **Type hints** for all functions
- **Docstrings** for all classes and methods
- **Error handling** with meaningful messages
- **Unit tests** for new features
- **Follow PEP 8** style guide

### 📚 Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples or use cases
- Improve API documentation
- Create tutorials or guides

## Code Review Process

1. **Automated checks** must pass (tests, linting, security scans)
2. **Manual review** by maintainers
3. **Testing** with different audio formats and configurations
4. **Performance** impact assessment
5. **Documentation** review

## Development Guidelines

### 🧪 Testing

- **Unit tests** for individual components
- **Integration tests** for full workflows
- **Docker tests** for containerised deployment
- **Performance tests** for large audio files

### 🔒 Security

- **No hardcoded secrets** or credentials
- **Input validation** for all user inputs
- **Secure handling** of audio files
- **Privacy considerations** for transcription data

### 📋 Pull Request Guidelines

**Title Format**: `[Type] Brief description`
- `[Feature]` - New functionality
- `[Fix]` - Bug fixes
- `[Docs]` - Documentation only
- `[Refactor]` - Code restructuring
- `[Test]` - Testing improvements

**PR Description Should Include**:
- Summary of changes
- Related issue number(s)
- Testing performed
- Breaking changes (if any)
- Screenshots (if UI changes)

### 🚀 Release Process

1. **Version bump** using semantic versioning
2. **Update CHANGELOG.md**
3. **Tag release** with version number
4. **GitHub release** with release notes
5. **Docker image** update

## Community Guidelines

- **Be respectful** and inclusive
- **Provide constructive feedback**
- **Help others** learn and contribute
- **Follow the Code of Conduct**

## Getting Help

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and community chat
- **Documentation** - Check the README and docs/ folder

## Recognition

Contributors will be recognised in:
- **README.md** acknowledgements
- **Release notes** for significant contributions
- **GitHub contributors** page

Thank you for helping make this project better! 🎉
