# Backend Setup Guide

Complete setup instructions for the AppSageAI backend service.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Memory**: 4GB RAM minimum
- **Storage**: 1GB free space
- **OS**: Linux, macOS, or Windows with WSL

### Required Accounts
- **Google Cloud Account** (free tier works)
- **Firebase Project** (uses same GCP project)
- **Groq API Key** ([Get here](https://console.groq.com))
- **HuggingFace Token** (optional, [Get here](https://huggingface.co/settings/tokens))

## 🔧 Installation Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/appsageai.git
cd appsageai/backend
```

### Step 2: Python Environment Setup

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Option B: Using UV Package Manager
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

#### Option C: Using Conda
```bash
# Create conda environment
conda create -n appsageai python=3.11

# Activate
conda activate appsageai
```

### Step 3: Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install test dependencies (optional)
pip install -r test_requirements.txt
```

If you encounter issues:
```bash
# Install one by one
pip install fastapi uvicorn
pip install firebase-admin google-cloud-firestore
pip install langchain langchain-groq langchain-huggingface
pip install pydantic pydantic-settings
pip install cryptography python-jose
```

### Step 4: Firebase Setup

#### 4.1 Create Firebase Project
1. Go to [Firebase Console](https://console.firebase.google.com)
2. Click "Create Project" or "Add Project"
3. Enter existing GCP project ID: `appsageai-472321`
4. Follow setup wizard

#### 4.2 Enable Authentication
1. In Firebase Console → Authentication → Get Started
2. Enable providers:
   - Email/Password
   - Google
   - Apple (requires Apple Developer account)

#### 4.3 Create Firestore Database
1. Firebase Console → Firestore Database → Create Database
2. Choose "Production mode"
3. Select location: `us-central1`
4. Database ID: `appsageai` (or `(default)`)

#### 4.4 Download Service Account Key
1. Firebase Console → Project Settings → Service Accounts
2. Click "Generate New Private Key"
3. Save as `backend/service-account.json`
4. **Never commit this file!**

### Step 5: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your values
nano .env  # or use any editor
```

Required configuration in `.env`:

```env
# Application
ENVIRONMENT=development
DEBUG=true

# Google Cloud
GCP_PROJECT_ID=appsageai-472321
FIRESTORE_DATABASE=appsageai  # or (default)
FIREBASE_SERVICE_ACCOUNT_PATH=./service-account.json

# API Keys
GROQ_API_KEY=gsk_xxxxxxxxxxxxx  # Get from https://console.groq.com
HF_TOKEN=hf_xxxxxxxxxxxxx  # Optional, from HuggingFace

# Security (generate these)
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this
ENCRYPTION_KEY=<generated-fernet-key>

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
```

Generate encryption key:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Step 6: Verify Setup

```bash
# Test imports
python -c "from app.main import app; print('✅ Setup successful!')"

# Run the test script
python test_backend.py
```

### Step 7: Run the Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🔍 Verification

### Check API Documentation
Open browser and navigate to:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

### Expected Health Response
```json
{
  "status": "healthy",
  "app": "AppSageAI",
  "version": "2.0.0",
  "environment": "development"
}
```

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Ensure virtual environment is activated
which python  # Should show .venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Firebase Authentication Error
```bash
# Check service account file exists
ls -la service-account.json

# Verify project ID matches
cat service-account.json | grep project_id
```

### Issue: Firestore Connection Error
```bash
# Check database name in .env
# Should match Firebase Console

# If database is named, not (default):
FIRESTORE_DATABASE=appsageai  # Your actual database name
```

### Issue: CORS Errors
```bash
# Update .env with correct origins
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]
```

### Issue: Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
uvicorn app.main:app --port 8001
```

## 🔒 Security Checklist

- [ ] `service-account.json` is in `.gitignore`
- [ ] `.env` is in `.gitignore`
- [ ] Generated new `JWT_SECRET_KEY`
- [ ] Generated new `ENCRYPTION_KEY`
- [ ] Updated `CORS_ORIGINS` for production
- [ ] Set `DEBUG=false` for production
- [ ] Set `ENVIRONMENT=production` for production

## 📊 Performance Optimization

### Development Settings
```env
DEBUG=true
RATE_LIMIT_ENABLED=false
```

### Production Settings
```env
DEBUG=false
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

### Running with Multiple Workers
```bash
# For production
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

## 🧪 Testing the Setup

### Run Unit Tests
```bash
pytest tests/test_auth.py -v
```

### Run Integration Tests
```bash
pytest tests/integration/ -v
```

### Test with cURL
```bash
# Health check
curl http://localhost:8000/health

# API root
curl http://localhost:8000/api/v1
```

## 📝 Next Steps

1. [Set up Postman](./POSTMAN_SETUP.md) for API testing
2. [Run the test suite](./TESTING_GUIDE.md)
3. [Set up the frontend](./FRONTEND_SETUP.md)
4. [Deploy to production](./DEPLOYMENT.md)

## 📚 Additional Resources

- [API Documentation](./API_DOCUMENTATION.md)
- [Architecture Overview](./ARCHITECTURE.md)
- [Contributing Guidelines](./CONTRIBUTING.md)

## 💡 Tips

- Use `python -m app.main` as alternative to `uvicorn`
- Enable SQL logging: `export LOG_LEVEL=DEBUG`
- Watch logs: `tail -f logs/running_logs.log`
- Use Postman collection for testing
- Keep dependencies updated: `pip list --outdated`

## 🆘 Getting Help

- Check [Issues](https://github.com/yourusername/appsageai/issues)
- Join [Discussions](https://github.com/yourusername/appsageai/discussions)
- Email: support@appsageai.com