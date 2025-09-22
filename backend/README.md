# AppSageAI Backend

FastAPI-based backend service for AppSageAI resume analysis platform.

## 🚀 Quick Start

```bash
# Install dependencies
uv init
uv install -r pyproject.toml

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Run server
uvicorn app.main:app --reload --port 8000
```

## 📁 Structure

```
backend/
├── app/
│   ├── api/          # API endpoints
│   ├── auth/         # Authentication
│   ├── core/         # Business logic
│   ├── db/           # Database models
│   ├── services/     # External services
│   ├── config/       # Configuration
│   └── main.py       # FastAPI app
├── tests/            # Test suite
└── logs/            # Application logs
```

## 🔧 Commands

### Development
```bash
# Run with auto-reload
uvicorn app.main:app --reload --port 8000

# Run specific port
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific tests
pytest tests/test_auth.py -v

# Run in watch mode
pip install pytest-watch
ptw
```

### Code Quality
```bash
# Format code
black app/

# Lint code
ruff check app/

# Type checking
mypy app/
```

## 📚 API Documentation

Once the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔐 Environment Variables

Required in `.env`:
```env
# Firebase/Google Cloud
GCP_PROJECT_ID=your-project-id
FIRESTORE_DATABASE=your-database-name
FIREBASE_SERVICE_ACCOUNT_PATH=./service-account.json

# API Keys
GROQ_API_KEY=your-groq-api-key

# Security
JWT_SECRET_KEY=generate-a-secret-key
ENCRYPTION_KEY=generate-with-fernet
```

Generate encryption key:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## 📝 Documentation

For detailed documentation, see:
- [Complete Setup Guide](../docs/BACKEND_SETUP.md)
- [API Documentation](../docs/API_DOCUMENTATION.md)
- [Testing Guide](../docs/TESTING_GUIDE.md)
- [Postman Setup](../docs/POSTMAN_SETUP.md)

## 🧪 Testing

```bash
# Install test dependencies
uv install -r pyproject.toml

# Run tests
pytest

# Generate coverage report
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## 🐛 Debugging

```bash
# Run with debug logs
export DEBUG=true
python -m app.main

# Check logs
tail -f logs/running_logs.log
```

## 📊 Performance

- Health check: < 100ms
- Analysis endpoints: < 2s
- File upload: supports up to 10MB

## 🔗 Related

- [Frontend README](../frontend/README.md)
- [Main README](../README.md)
- [Contributing](../docs/CONTRIBUTING.md)