# Testing Guide for AppSageAI

## 📋 Overview

This guide covers all testing approaches for AppSageAI backend:
- Unit Tests
- Integration Tests
- API Testing with Postman
- Load Testing
- Manual Testing

## 🛠️ Setup Testing Environment

### Prerequisites

```bash
# Navigate to backend
cd backend

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install test dependencies
uv install -r pyproject.toml
```

### Environment Configuration

Create `.env.test`:
```env
ENVIRONMENT=testing
DEBUG=true
RATE_LIMIT_ENABLED=false
GCP_PROJECT_ID=test-project
FIRESTORE_DATABASE=test-db
GROQ_API_KEY=test-key
```

## 🧪 Running Tests

### All Tests
```bash
pytest
```

### With Coverage Report
```bash
pytest --cov=app --cov-report=html --cov-report=term
# View HTML report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

### Specific Test Categories

#### Unit Tests Only
```bash
pytest -m unit
```

#### Integration Tests Only
```bash
pytest -m integration
```

#### By Module
```bash
# Authentication tests
pytest tests/test_auth.py -v

# Resume management tests
pytest tests/test_resume.py -v

# Chat tests
pytest tests/test_chat.py -v

# Analysis tests
pytest tests/test_analysis.py -v
```

### Quick Tests (Skip Slow)
```bash
pytest -m "not slow"
```

### Failed Tests Only
```bash
# Run only previously failed tests
pytest --lf

# Run failed first, then others
pytest --ff
```

### Verbose Output
```bash
pytest -vv
```

### Watch Mode (Auto-run on changes)
```bash
# Install pytest-watch first
pip install pytest-watch

# Run in watch mode
ptw
```

### Parallel Execution
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto  # Auto-detect CPU cores
pytest -n 4     # Use 4 workers
```

## 📊 Test Structure

### Directory Layout
```
backend/tests/
├── conftest.py           # Shared fixtures
├── test_auth.py         # Authentication tests
├── test_resume.py       # Resume management tests
├── test_chat.py         # Chat session tests
├── test_analysis.py     # Analysis engine tests
├── test_user.py         # User management tests
├── test_encryption.py   # Encryption service tests
└── integration/         # Integration tests
    ├── test_api_flow.py # Full API workflow tests
    └── test_firebase.py # Firebase integration tests
```

### Writing Tests

#### Basic Test Structure
```python
import pytest
from unittest.mock import Mock, patch

class TestFeatureName:
    """Test suite for feature."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture for test data."""
        return {"key": "value"}
    
    def test_success_case(self, sample_data):
        """Test successful operation."""
        # Arrange
        expected = "expected_result"
        
        # Act
        result = function_under_test(sample_data)
        
        # Assert
        assert result == expected
    
    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_under_test(invalid_data)
    
    @patch('app.module.external_service')
    def test_with_mock(self, mock_service):
        """Test with mocked dependency."""
        mock_service.return_value = "mocked_response"
        
        result = function_under_test()
        
        assert result == "expected"
        mock_service.assert_called_once()
```

#### Async Test Example
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result == expected
```

## 🔍 Test Categories

### Unit Tests
- Test individual functions/methods
- Mock all external dependencies
- Fast execution
- High code coverage

```python
@pytest.mark.unit
def test_hash_identifier():
    """Test identifier hashing."""
    from app.services.encryption import encryption_service
    
    email = "test@example.com"
    hashed = encryption_service.hash_identifier(email)
    
    assert len(hashed) == 16
    assert hashed != email
```

### Integration Tests
- Test component interactions
- Use test database
- Mock external APIs only
- Medium execution time

```python
@pytest.mark.integration
async def test_full_analysis_flow():
    """Test complete analysis workflow."""
    # Upload resume
    # Create chat
    # Run analysis
    # Verify results
```

### End-to-End Tests
- Test complete user journeys
- Real Firebase (test project)
- Real services
- Slow execution

```python
@pytest.mark.e2e
@pytest.mark.slow
async def test_complete_user_journey():
    """Test from signup to analysis."""
    # Create user
    # Upload resume
    # Create multiple chats
    # Run analyses
    # Check limits
```

## 📈 Coverage Requirements

### Target Coverage
- Overall: **80%+**
- Core modules: **90%+**
- API endpoints: **95%+**
- Utilities: **70%+**

### View Coverage Report
```bash
# Generate report
pytest --cov=app --cov-report=html

# Check specific module
pytest --cov=app.api.auth tests/test_auth.py

# Show missing lines
pytest --cov=app --cov-report=term-missing
```

## 🐛 Debugging Tests

### Run Single Test
```bash
pytest tests/test_auth.py::TestAuthEndpoints::test_verify_token_new_user -v
```

### Show Print Statements
```bash
pytest -s  # No capture
```

### Drop into Debugger on Failure
```bash
pytest --pdb
```

### Show Local Variables on Failure
```bash
pytest -l
```

### Maximum Verbosity
```bash
pytest -vvv --tb=long
```

## 🚦 CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r test_requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## 🔄 Load Testing

### Using Locust

Create `locustfile.py`:
```python
from locust import HttpUser, task, between

class AppSageUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login and get token."""
        self.token = "your-test-token"
        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }
    
    @task
    def health_check(self):
        self.client.get("/health")
    
    @task(3)
    def list_chats(self):
        self.client.get(
            "/api/v1/chat/list",
            headers=self.headers
        )
    
    @task(2)
    def get_stats(self):
        self.client.get(
            "/api/v1/user/stats",
            headers=self.headers
        )
```

Run load test:
```bash
# Install Locust
pip install locust

# Run test
locust -f locustfile.py --host=http://localhost:8000

# Open browser
open http://localhost:8089
```

## 📝 Test Documentation

### Document Test Purpose
```python
def test_resume_upload_size_limit():
    """
    Test that resume upload enforces file size limit.
    
    Requirements:
    - File size must be under 10MB
    - Should return 400 for larger files
    - Should include error message
    """
```

### Use Descriptive Names
```python
# Good
def test_expired_token_returns_401_unauthorized():

# Bad  
def test_auth_fail():
```

### Group Related Tests
```python
class TestResumeUpload:
    """Tests for resume upload functionality."""
    
    class TestValidation:
        """Validation tests."""
        
    class TestStorage:
        """Storage tests."""
```

## ✅ Testing Checklist

Before committing:

- [ ] All tests pass
- [ ] Coverage > 80%
- [ ] No skipped tests without reason
- [ ] New features have tests
- [ ] Bug fixes have regression tests
- [ ] Integration tests pass
- [ ] No hardcoded test data
- [ ] Mocks are properly cleaned up
- [ ] No test interdependencies

## 🎯 Best Practices

1. **Independent Tests**: Each test should run independently
2. **Clear Names**: Test names should describe what they test
3. **Single Assertion**: Prefer one assertion per test
4. **Mock External**: Always mock external services
5. **Test Data**: Use fixtures for reusable test data
6. **Clean Up**: Always clean up resources
7. **Fast Tests**: Keep unit tests under 1 second
8. **Deterministic**: Tests should always produce same results
9. **Documentation**: Document complex test scenarios
10. **Coverage**: Aim for quality over quantity

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Add project to path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

#### Firebase Errors
```bash
# Use mock in tests
export ENVIRONMENT=testing
```

#### Async Test Errors
```bash
# Install async support
pip install pytest-asyncio
```

#### Coverage Not Working
```bash
# Reinstall coverage
pip install --upgrade pytest-cov
```

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://testdriven.io/blog/testing-python/)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Postman Testing](./POSTMAN_SETUP.md)