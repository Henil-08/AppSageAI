# Contributing to AppSageAI

Thank you for your interest in contributing to AppSageAI! We're building a privacy-first resume analysis platform that helps job seekers succeed, and we'd love your help.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect user privacy above all

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- UV package manager (or pip)
- Google Cloud account (free tier works)
- Firebase project setup

### Setup Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/appsageai.git
   cd appsageai
   ```

2. **Backend Setup**
   ```bash
   cd backend
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Configure Environment**
   - Copy `.env.example` to `.env`
   - Add your API keys
   - Set up Firebase service account

4. **Run Tests**
   ```bash
   ./run_tests.sh
   ```

## 📝 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Follow the existing code structure
- Add tests for new features
- Update documentation if needed

### 3. Code Style

We use:
- **Black** for Python formatting
- **Ruff** for linting
- **MyPy** for type checking

Run formatters:
```bash
black app/
ruff check app/
mypy app/
```

### 4. Testing

Write tests for your changes:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=app
```

### 5. Commit Messages

Follow conventional commits:
```
feat: add new analysis type for skills
fix: correct token counting in analyzer
docs: update API documentation
test: add tests for chat endpoints
refactor: simplify encryption service
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 🏗️ Architecture Guidelines

### Backend Structure

```
backend/
├── app/
│   ├── api/          # API endpoints
│   ├── auth/         # Authentication
│   ├── core/         # Business logic
│   ├── db/           # Database models
│   └── services/     # External services
```

### Key Principles

1. **Privacy First**
   - Never log PII
   - Encrypt sensitive data
   - Use hashed identifiers

2. **Clean Code**
   - Single responsibility
   - Clear function names
   - Comprehensive docstrings

3. **Error Handling**
   - Always handle exceptions
   - Return meaningful errors
   - Log errors appropriately

## 📚 Documentation

### API Documentation

- Update `API_README.md` for new endpoints
- Include request/response examples
- Document error cases

### Code Documentation

```python
def analyze_resume(
    self,
    resume_content: bytes,
    job_description: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Analyze resume against job description.
    
    Args:
        resume_content: Resume file content
        job_description: Job description text
    
    Returns:
        Tuple of (analysis_result, metadata)
    
    Raises:
        ValueError: If resume content is invalid
    """
```

## 🧪 Testing Requirements

### Test Coverage

- Aim for >80% code coverage
- Test both success and error cases
- Include edge cases

### Test Structure

```python
class TestFeatureName:
    """Test suite for feature."""
    
    def test_success_case(self):
        """Test successful operation."""
        # Arrange
        # Act  
        # Assert
    
    def test_error_case(self):
        """Test error handling."""
        # Test exception handling
```

## 🐛 Reporting Issues

### Bug Reports

Include:
1. Description of the bug
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details

### Feature Requests

Include:
1. Problem it solves
2. Proposed solution
3. Alternative solutions
4. Additional context

## 🔒 Security

### Reporting Security Issues

**DO NOT** create public issues for security vulnerabilities.

Email: security@appsageai.com

Include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## 📋 Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No sensitive data in commits
- [ ] Meaningful commit messages
- [ ] PR description explains changes

## 🎯 Areas for Contribution

### High Priority

- [ ] Frontend development (Next.js)
- [ ] Additional analysis types
- [ ] Performance optimizations
- [ ] Documentation improvements
- [ ] Test coverage increase

### Good First Issues

Look for issues labeled `good first issue` on GitHub.

### Feature Ideas

- Multi-language support
- Additional LLM providers
- Resume templates
- Interview preparation
- Salary negotiation tips

## 💬 Getting Help

- **Discord**: [Join our server](https://discord.gg/appsageai)
- **Discussions**: GitHub Discussions
- **Email**: contributors@appsageai.com

## 🙏 Recognition

Contributors will be:
- Listed in AUTHORS.md
- Mentioned in release notes
- Invited to contributor meetings

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making AppSageAI better! 🚀