# AppSageAI Backend

FastAPI-based backend service for AppSageAI - Privacy-first AI Resume Analysis Platform.

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/appsageai.git
cd appsageai/backend

# Install UV package manager (any method is fine)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv init
uv pip install -r pyproject.toml

# Setup environment
cp .env.example .env
# Edit .env with your API keys and configuration

# Run server
uvicorn app.main:app --reload --port 8000
```

## Project Structure

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   ├── analysis.py   # Resume analysis endpoints
│   │   ├── auth.py       # Authentication endpoints
│   │   ├── chat.py       # Chat session management
│   │   ├── prompts.py    # Custom prompts management
│   │   ├── resume.py     # Resume upload/management
│   │   └── user.py       # User management
│   ├── auth/             # Firebase authentication
│   ├── core/             # Business logic
│   │   ├── resume_analyzer.py  # RAG-based analysis
│   │   ├── prompts.py          # Prompt templates
│   │   └── job_extractor.py    # Job details extraction
│   ├── db/               # Database models
│   ├── services/         # External services
│   │   └── encryption.py # AES-256 encryption
│   ├── config/           # Configuration
│   │   └── settings.py   # Environment settings
│   ├── logger.py         # Logging configuration
│   └── main.py           # FastAPI application
├── tests/                # Test suite
├── logs/                 # Application logs
├── service-account.json  # Firebase credentials (git-ignored)
├── pyproject.toml        # Python dependencies
└── .env                  # Environment variables (git-ignored)
```

## Development Commands

### Running the Server

```bash
# Development with auto-reload
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# With custom settings
uvicorn app.main:app --reload --port 8080 --log-level debug
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
open htmlcov/index.html  # View coverage report

# Run specific test file
pytest tests/test_auth.py -v

# Run in watch mode
pip install pytest-watch
ptw

# Run only fast tests
pytest -m "not slow"
```

### Code Quality

```bash
# Format code with Black
black app/

# Lint with Ruff
ruff check app/
ruff check app/ --fix  # Auto-fix issues

# Type checking with MyPy
mypy app/

# Run all checks
black app/ && ruff check app/ && mypy app/
```

## Environment Configuration

Create a `.env` file with the following variables:

```env
# Application Settings
APP_NAME=AppSageAI
APP_VERSION=2.0.0
ENVIRONMENT=development
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1

# Firebase/Google Cloud
GCP_PROJECT_ID=your-project-id
FIREBASE_SERVICE_ACCOUNT_PATH=./service-account.json
FIRESTORE_DATABASE=your-database-name

# AI Models (You can use any models)
MODEL_NAME=gemini-2.5-flash
JOB_MODEL_NAME=gemini-2.5-flash-lite
EMBEDDING_MODEL=all-MiniLM-L6-v2

# HuggingFace (for embeddings)
HF_TOKEN=hf_xxxxxxxxxxxxx

# Security - IMPORTANT: Generate these!
JWT_SECRET_KEY=your-super-secret-key-change-this
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=30

# Encryption - Generate using command below
ENCRYPTION_KEY=

# CORS Origins
CORS_ORIGIN_1=http://localhost:3000
CORS_ORIGIN_2=https://your-frontend-domain.com
CORS_ORIGIN_3=

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60

# Document Processing
MAX_FILE_SIZE_MB=10
ALLOWED_FILE_TYPE=pdf
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Generate Security Keys

```bash
# Generate encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Generate JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## API Documentation

Once the server is running, access interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/health

### Main API Endpoints

- `/api/v1/auth/*` - Authentication (Firebase token verification)
- `/api/v1/resume/*` - Resume upload and management
- `/api/v1/chat/*` - Chat session management
- `/api/v1/analysis/*` - AI-powered resume analysis
- `/api/v1/prompts/*` - Custom prompt templates
- `/api/v1/user/*` - User profile and stats

## Testing Guide

```bash
# Install test dependencies
uv pip install pytest pytest-cov pytest-asyncio pytest-watch

# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage report
pytest --cov=app --cov-report=term-missing

# Run specific test
pytest tests/test_auth.py::TestAuthEndpoints::test_verify_token -v

# Debug mode (show print statements)
pytest -s

# Run tests in parallel
pip install pytest-xdist
pytest -n auto
```

## Debugging

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with Python debugger
python -m pdb -m uvicorn app.main:app --reload

# Check application logs
tail -f logs/running_logs.log

# Test specific endpoint with cURL
curl -X GET http://localhost:8000/health
```

## Performance Monitoring

### Expected Response Times
- Health check: < 100ms
- Authentication: < 200ms
- Resume upload: < 2s
- Analysis endpoints: < 3s (streaming)
- Chat operations: < 500ms

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Open browser to http://localhost:8089
```

## Security Features

- **AES-256 Encryption**: All user data encrypted at rest
- **Firebase Authentication**: Secure token-based auth
- **Rate Limiting**: Protection against abuse
- **CORS Protection**: Restricted origins
- **Input Validation**: Pydantic models for all inputs
- **SQL Injection Prevention**: Using Firestore NoSQL
- **XSS Protection**: Input sanitization
- **Secure Headers**: Security middleware

## Troubleshooting

### Common Issues

**ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
which python  # Should show .venv/bin/python
# Reinstall dependencies
uv pip install -r pyproject.toml
```

**Firebase Authentication Error**
```bash
# Check service account file
ls -la service-account.json
# Verify project ID in .env matches Firebase Console
```

**Port Already in Use**
```bash
# Find process using port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
uvicorn app.main:app --port 8001
```

## 🔗 Related Documentation

- [Complete Setup Guide](../docs/SETUP.md)
- [API Documentation](../docs/API_DOCUMENTATION.md)
- [Frontend README](../frontend/README.md)
- [Main README](../README.md)

## 📧 Support

For issues or questions:
- Create an issue on GitHub
- Check existing issues for solutions
- Review the complete setup guide in docs/

---

**Built with FastAPI, Firebase, and Google Gemini AI**