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

Instead, please send an email to: [security@example.com] with the following information:

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

### 🚨 Common Security Risks

#### Input Validation
- **Audio File Validation**: Validate audio file formats and sizes
- **Configuration Validation**: Sanitise configuration inputs
- **Path Traversal**: Prevent directory traversal attacks

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
- **Email**: [security@example.com]
- **PGP Key**: [Link to PGP key if available]

Thank you for helping keep our project secure! 🔐
