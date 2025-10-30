# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability, please report it responsibly:

### 🔒 Private Reporting

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities privately using GitHub's Security Advisory feature:

1. Go to the repository's "Security" tab
2. Click "Report a vulnerability"
3. Fill in the details

Or open a private issue with the following information:

- **Vulnerability Description**: Clear description of the issue
- **Steps to Reproduce**: Detailed steps to reproduce the vulnerability
- **Impact Assessment**: Potential impact and severity
- **Suggested Fix**: If you have ideas for fixing the issue
- **Your Contact Info**: For follow-up questions

### 📋 What to Expect

- **Initial Response**: Within 48 hours
- **Investigation**: We'll investigate and assess the issue
- **Updates**: Regular updates on progress every 5 business days
- **Resolution**: Security fixes will be prioritised and released promptly
- **Credit**: We'll acknowledge your contribution (if desired)

## Security Considerations

### 🎵 Audio File Handling

- **Local Processing**: All audio processing happens locally
- **Temporary Files**: Temporary files are cleaned up after processing
- **File Permissions**: Ensure proper file permissions on audio files
- **No External Upload**: Audio files are never uploaded to external services

### 🔐 Data Privacy

- **Local LLM**: Uses local Ollama instance, no data sent to external APIs
- **Transcription Data**: All transcription data remains on your system
- **No Logging**: Sensitive conversation content is not logged
- **Memory Cleanup**: Audio data is cleared from memory after processing

### 🐳 Docker Security

- **Non-Root User**: Docker containers run as non-root user
- **Isolated Network**: Services run in isolated Docker network
- **Volume Permissions**: Proper volume permissions and mounting
- **Image Security**: Regular base image updates for security patches

### 🛡️ Implemented Security Protections

#### Path Traversal Protection (v1.0+)
- ✅ **Directory Traversal Prevention**: All user-supplied file paths are validated
- ✅ **Whitelisted Directories**: Access restricted to project directories only
- ✅ **Extension Validation**: Only approved file extensions allowed (.mp3, .wav, .txt, etc.)
- Files outside the project directory are rejected with clear error messages

#### Resource Limits (v1.0+)
- ✅ **Audio File Size Limit**: Maximum 500MB per audio file
- ✅ **Prompt File Size Limit**: Maximum 1MB per prompt file
- ✅ **Column Count Limit**: Maximum 50 columns in structured output
- Prevents denial-of-service attacks via resource exhaustion

#### Input Validation
- ✅ **Audio File Validation**: File format and size validation before processing
- ✅ **Configuration Validation**: Configuration inputs are sanitized
- ✅ **Filename Sanitization**: Basenames extracted to prevent path injection

#### Network Security
- **Local Services**: Ollama and application run on localhost only
- **Port Exposure**: Only expose necessary ports
- **Service Authentication**: Use proper authentication for services

#### Dependency Security
- **Regular Updates**: Keep dependencies updated
- **Vulnerability Scanning**: Monitor for known vulnerabilities
- **Minimal Dependencies**: Use only necessary dependencies

### 🔧 Security Best Practices

#### For Users
- **File Permissions**: Set appropriate permissions on audio files
- **Network Isolation**: Run in isolated network environment
- **Regular Updates**: Keep the application and dependencies updated
- **Backup Security**: Secure backups of transcription data

#### For Developers
- **Code Review**: All code changes require security review
- **Secret Management**: Never commit secrets or credentials
- **Input Sanitisation**: Validate and sanitise all inputs
- **Error Handling**: Don't expose sensitive information in errors

### 🛡️ Security Features

- **No External Dependencies**: All processing happens locally
- **Encrypted Communication**: HTTPS for any web interfaces
- **Secure Defaults**: Secure configuration defaults
- **Audit Logging**: Security events are logged appropriately

### 🔍 Security Testing

We regularly perform:
- **Static Code Analysis**: Automated security scanning
- **Dependency Scanning**: Monitor for vulnerable dependencies
- **Container Scanning**: Docker image vulnerability scanning
- **Penetration Testing**: Regular security assessments

#### Testing Security Protections

You can verify the implemented security protections:

**Path Traversal Protection:**
```bash
# These should fail with security errors
python transcribe_and_summarise.py --file ../../etc/passwd
python transcribe_and_summarise.py --structured ../../../../tmp/malicious.txt
python transcribe_and_summarise.py --config ../../../sensitive.conf
```

**File Size Limits:**
```bash
# Create a large file (should be rejected)
dd if=/dev/zero of=large_test.mp3 bs=1M count=600
python transcribe_and_summarise.py --file large_test.mp3
```

**Valid Usage (should work):**
```bash
# These should work normally
python transcribe_and_summarise.py --file activitycall.mp3
python transcribe_and_summarise.py --structured structured_prompt_example.txt
```

## Responsible Disclosure

We believe in responsible disclosure and will work with security researchers to:

1. **Acknowledge Receipt**: Confirm we've received your report
2. **Investigate Thoroughly**: Assess the issue and potential impact
3. **Develop Fix**: Create and test a security fix
4. **Coordinate Release**: Plan the release timing
5. **Public Disclosure**: Publish details after fix is available

## Security Updates

Security updates will be:
- **Prioritised**: Given highest priority for development
- **Backported**: Applied to supported versions
- **Documented**: Included in security advisories
- **Communicated**: Announced through appropriate channels

## Contact

For security-related questions or concerns:
- **GitHub Security Advisories**: Use the repository's Security tab
- **GitHub Issues**: For general security questions (not vulnerabilities)

Thank you for helping keep our project secure! 🔐
