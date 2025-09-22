# Contributing to AppSageAI

Thank you for your interest in contributing to AppSageAI! We're building a privacy-first resume analysis platform that helps job seekers succeed.

## 🤝 Code of Conduct

By participating in this project, you agree to:
- Be respectful and inclusive
- Welcome newcomers and help them get started  
- Focus on constructive criticism
- Respect user privacy above all

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ with UV package manager
- Node.js 18+ with npm
- Google Cloud account (free tier works)
- Git and GitHub account

### Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/YOUR_USERNAME/appsageai.git
cd appsageai
```

2. **Setup Instructions**
- **Backend Setup**: Follow [backend/README.md](./backend/README.md#-quick-start)
- **Frontend Setup**: Follow [frontend/README.md](./frontend/README.md#-quick-start)
- **Full GCP Setup**: See [docs/SETUP.md](./docs/SETUP.md)

## 📝 How to Contribute

### 1. Find an Issue
- Check [open issues](https://github.com/appsageai/appsageai/issues)
- Look for `good first issue` labels
- Comment on the issue to claim it

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Development Guidelines

#### Code Style
- **Python**: Black formatter, Ruff linter, MyPy types
- **TypeScript**: ESLint, Prettier
- **Commits**: Conventional commits format

```bash
# Python formatting
cd backend
black app/
ruff check app/
mypy app/

# TypeScript formatting  
cd frontend
npm run lint
npm run format
```

#### Commit Messages
```
feat: add resume tagging feature
fix: resolve streaming timeout issue
docs: update deployment guide
test: add chat endpoint tests
refactor: simplify encryption service
chore: update dependencies
```

### 4. Testing

#### Backend Testing
```bash
cd backend
pytest                        # Run all tests
pytest --cov=app             # With coverage
pytest tests/test_auth.py    # Specific file
```

#### Frontend Testing
```bash
cd frontend
npm test                     # Run tests
npm run test:coverage       # Coverage report
```

### 5. Submit Pull Request

1. Push your branch
2. Open PR against `main` branch
3. Fill out PR template
4. Wait for review

## 🏗️ Architecture Overview

For detailed architecture, see:
- [Project Structure](./docs/SETUP.md#-project-structure)
- [Backend Architecture](./backend/README.md#-project-structure)
- [Frontend Architecture](./frontend/README.md#-project-structure)

### Key Design Principles

#### Privacy First
- Encrypt all sensitive data (AES-256)
- No PII in logs or analytics
- Hash user identifiers
- Minimize data collection

#### Clean Code
- Single responsibility principle
- Clear, descriptive naming
- Comprehensive documentation
- Type safety (TypeScript/Python types)

#### Error Handling
```python
# Always handle exceptions gracefully
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    return error_response(detail="User-friendly message")
```

## 📚 Documentation Standards

### Code Documentation

#### Python Docstrings
```python
def analyze_resume(
    self,
    resume_content: bytes,
    job_description: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Analyze resume against job description using RAG.
    
    Args:
        resume_content: PDF file content as bytes
        job_description: Target job description text
    
    Returns:
        Tuple of (analysis_result, metadata_dict)
    
    Raises:
        ValueError: If resume content is invalid
        
    Example:
        result, meta = analyzer.analyze_resume(pdf_bytes, "Senior Engineer...")
    """
```

#### TypeScript JSDoc
```typescript
/**
 * Fetches chat session with all messages
 * @param sessionId - Unique chat session identifier
 * @returns Promise with chat data or null if not found
 * @throws {ApiError} If request fails
 */
async function getChatSession(sessionId: string): Promise<ChatSession | null> {
  // Implementation
}
```

### API Documentation
When adding/modifying endpoints:
1. Update OpenAPI schema
2. Add request/response examples
3. Document error codes
4. Update [API_DOCUMENTATION.md](./docs/API_DOCUMENTATION.md)

## 🧪 Testing Requirements

### Coverage Standards
- **Overall**: Minimum 80% coverage
- **New features**: Must include tests
- **Bug fixes**: Must include regression test

### Test Structure
```python
class TestFeatureName:
    """Test suite for specific feature."""
    
    @pytest.fixture
    def sample_data(self):
        """Reusable test data."""
        return {"key": "value"}
    
    def test_success_case(self, sample_data):
        """Test normal operation."""
        # Arrange
        expected = "expected_result"
        
        # Act  
        result = function_under_test(sample_data)
        
        # Assert
        assert result == expected
    
    def test_error_handling(self):
        """Test error scenarios."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

## 🔒 Security Guidelines

### Handling Sensitive Data
- Never log passwords, tokens, or PII
- Use encryption service for storage
- Sanitize user input
- Validate all inputs with Pydantic/Zod

### Reporting Security Issues
**DO NOT** open public issues for vulnerabilities.

Email: henilgajjar08@gmail.com

Include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

## 📋 Pull Request Checklist

Before submitting:
- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] Added/updated tests for changes
- [ ] Documentation updated if needed
- [ ] No sensitive data in commits
- [ ] Self-review completed
- [ ] PR description explains changes clearly

## 🎯 Priority Areas

### High Priority Features
- [ ] Multi-language support
- [ ] Additional AI providers (OpenAI, Anthropic)
- [ ] Real-time collaboration
- [ ] Advanced analytics dashboard
- [ ] Resume templates

### Technical Improvements
- [ ] Performance optimization
- [ ] Test coverage increase
- [ ] Documentation updates
- [ ] Accessibility improvements
- [ ] Mobile app development

### Good First Issues
- Adding loading states
- Improving error messages
- Writing tests
- Documentation fixes
- UI/UX improvements

## 💬 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and features
- **Discussions**: General questions
- **Email**: henilgajjar08@gmail.com

### Resources
- [Development Setup](./docs/SETUP.md)
- [API Documentation](./docs/API_DOCUMENTATION.md)
- [Deployment Guide](./docs/CICD_WORKFLOW.md)
- [Privacy Architecture](./docs/PRIVACY_ARCHITECTURE.md)

## 🙏 Recognition

Contributors are:
- Listed in [AUTHORS.md](./AUTHORS.md)
- Mentioned in release notes
- Given credit in documentation

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).

---

**Thank you for helping make AppSageAI better! 🚀**