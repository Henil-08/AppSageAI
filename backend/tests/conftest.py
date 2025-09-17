"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import Generator, Any
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client() -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_firebase_app():
    """Mock Firebase app initialization."""
    with patch('app.auth.firebase.firebase_admin.initialize_app') as mock_app:
        mock_app.return_value = MagicMock()
        yield mock_app


@pytest.fixture
def mock_firestore_client():
    """Mock Firestore client."""
    with patch('app.auth.firebase.get_firestore_client') as mock_client:
        mock_db = MagicMock()
        mock_client.return_value = mock_db
        yield mock_db


@pytest.fixture
def mock_auth_token():
    """Mock authenticated user token."""
    return {
        "uid": "test_uid_123",
        "email": "test@example.com",
        "email_verified": True,
        "name": "Test User",
        "firebase": {
            "sign_in_provider": "google.com"
        }
    }


@pytest.fixture
def auth_headers():
    """Mock authorization headers."""
    return {"Authorization": "Bearer mock_token_123"}


@pytest.fixture
def mock_verify_token(mock_auth_token):
    """Mock token verification."""
    with patch('app.auth.firebase.auth.verify_id_token') as mock_verify:
        mock_verify.return_value = mock_auth_token
        yield mock_verify


@pytest.fixture
def sample_resume_pdf():
    """Create a sample PDF file for testing."""
    content = b"%PDF-1.4\n%Test PDF content for resume"
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


@pytest.fixture
def sample_job_description():
    """Sample job description for testing."""
    return """
    We are looking for a Senior Full Stack Developer to join our team.
    
    Requirements:
    - 5+ years of experience with Python and JavaScript
    - Experience with React and FastAPI
    - Strong understanding of cloud platforms (AWS, GCP)
    - Experience with databases (PostgreSQL, MongoDB)
    
    Responsibilities:
    - Design and develop scalable web applications
    - Collaborate with cross-functional teams
    - Mentor junior developers
    """


@pytest.fixture
def mock_groq_llm():
    """Mock Groq LLM."""
    with patch('app.core.resume_analyzer.ChatGroq') as mock_llm:
        instance = MagicMock()
        instance.invoke.return_value = {
            "answer": "Mock analysis result",
            "context": ["context1", "context2"]
        }
        mock_llm.return_value = instance
        yield instance


@pytest.fixture
def mock_embeddings():
    """Mock HuggingFace embeddings."""
    with patch('app.core.resume_analyzer.HuggingFaceEmbeddings') as mock_emb:
        instance = MagicMock()
        mock_emb.return_value = instance
        yield instance


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    with patch('app.config.settings.settings') as mock_settings:
        mock_settings.app_name = "AppSageAI Test"
        mock_settings.environment = "testing"
        mock_settings.debug = True
        mock_settings.gcp_project_id = "test-project"
        mock_settings.firestore_database = "test-db"
        mock_settings.groq_api_key = "test-key"
        mock_settings.model_name = "test-model"
        mock_settings.cors_origins = ["http://localhost:3000"]
        mock_settings.rate_limit_enabled = False
        mock_settings.max_file_size_mb = 10
        mock_settings.get_groq_api_key.return_value = "test-key"
        yield mock_settings


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset Firebase app
    import app.auth.firebase as fb
    fb._firebase_app = None
    fb._firestore_client = None
    
    yield
    
    # Cleanup after test
    fb._firebase_app = None
    fb._firestore_client = None


@pytest.fixture
def mock_encryption_service():
    """Mock encryption service."""
    with patch('app.services.encryption.encryption_service') as mock_enc:
        mock_enc.generate_id.return_value = "test_id_123"
        mock_enc.hash_identifier.return_value = "hashed_123"
        mock_enc.encrypt_metadata.return_value = "encrypted_data"
        mock_enc.decrypt_metadata.return_value = "decrypted_data"
        yield mock_enc