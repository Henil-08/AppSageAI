"""Tests for authentication endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.main import app
from app.auth.firebase import get_current_user, get_current_user_uid


client = TestClient(app)


@pytest.fixture
def mock_firebase_token():
    """Mock Firebase ID token."""
    return {
        "uid": "test_user_123",
        "email": "test@example.com",
        "email_verified": True,
        "name": "Test User",
        "picture": "https://example.com/photo.jpg",
        "firebase": {
            "sign_in_provider": "google.com"
        },
        "auth_time": 1234567890
    }


@pytest.fixture
def auth_headers():
    """Mock authorization headers."""
    return {"Authorization": "Bearer mock_token_123"}


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    @patch('app.auth.firebase.auth.verify_id_token')
    @patch('app.auth.firebase.get_firestore_client')
    def test_verify_token_new_user(self, mock_firestore, mock_verify, mock_firebase_token):
        """Test token verification for new user."""
        # Setup mocks
        mock_verify.return_value = mock_firebase_token
        
        # Mock Firestore
        mock_db = MagicMock()
        mock_firestore.return_value = mock_db
        
        # Mock user doesn't exist
        mock_user_doc = MagicMock()
        mock_user_doc.exists = False
        mock_db.collection().document().get.return_value = mock_user_doc
        
        # Make request
        response = client.post(
            "/api/v1/auth/verify",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["uid"] == "test_user_123"
        assert data["email"] == "test@example.com"
        assert data["plan"] == "free"
    
    @patch('app.auth.firebase.auth.verify_id_token')
    @patch('app.auth.firebase.get_firestore_client')
    def test_verify_token_existing_user(self, mock_firestore, mock_verify, mock_firebase_token):
        """Test token verification for existing user."""
        # Setup mocks
        mock_verify.return_value = mock_firebase_token
        
        # Mock Firestore with existing user
        mock_db = MagicMock()
        mock_firestore.return_value = mock_db
        
        mock_user_doc = MagicMock()
        mock_user_doc.exists = True
        mock_user_doc.to_dict.return_value = {
            "uid": "test_user_123",
            "email": "test@example.com",
            "created_at": datetime.utcnow(),
            "plan": "pro",
            "usage": {
                "analyses_count": 50,
                "tokens_used": 75000
            }
        }
        mock_db.collection().document().get.return_value = mock_user_doc
        
        # Make request
        response = client.post(
            "/api/v1/auth/verify",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["plan"] == "pro"
        assert data["usage"]["analyses_count"] == 50
    
    def test_verify_token_invalid(self):
        """Test token verification with invalid token."""
        with patch('app.auth.firebase.auth.verify_id_token') as mock_verify:
            from firebase_admin import auth
            mock_verify.side_effect = auth.InvalidIdTokenError("Invalid token")
            
            response = client.post(
                "/api/v1/auth/verify",
                headers={"Authorization": "Bearer invalid_token"}
            )
            
            assert response.status_code == 401
            assert "Invalid authentication token" in response.json()["detail"]
    
    def test_verify_token_expired(self):
        """Test token verification with expired token."""
        with patch('app.auth.firebase.auth.verify_id_token') as mock_verify:
            from firebase_admin import auth
            mock_verify.side_effect = auth.ExpiredIdTokenError("Token expired")
            
            response = client.post(
                "/api/v1/auth/verify",
                headers={"Authorization": "Bearer expired_token"}
            )
            
            assert response.status_code == 401
            assert "Token has expired" in response.json()["detail"]
    
    @patch('app.auth.firebase.auth.verify_id_token')
    @patch('app.auth.firebase.get_firestore_client')
    def test_get_current_user(self, mock_firestore, mock_verify, mock_firebase_token):
        """Test getting current user profile."""
        # Setup mocks
        mock_verify.return_value = mock_firebase_token
        
        # Mock Firestore
        mock_db = MagicMock()
        mock_firestore.return_value = mock_db
        
        mock_user_doc = MagicMock()
        mock_user_doc.exists = True
        mock_user_doc.to_dict.return_value = {
            "uid": "test_user_123",
            "email": "test@example.com",
            "created_at": datetime.utcnow(),
            "plan": "free",
            "usage": {}
        }
        mock_db.collection().document().get.return_value = mock_user_doc
        
        # Make request
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["uid"] == "test_user_123"
        assert data["email"] == "test@example.com"
    
    def test_auth_check_no_token(self):
        """Test auth check without token."""
        response = client.get("/api/v1/auth/check")
        assert response.status_code == 403  # No bearer token provided
    
    @patch('app.auth.firebase.auth.verify_id_token')
    def test_logout(self, mock_verify, mock_firebase_token):
        """Test logout endpoint."""
        mock_verify.return_value = mock_firebase_token
        
        response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        assert "Logged out successfully" in response.json()["message"]


class TestAuthHelpers:
    """Test authentication helper functions."""
    
    def test_hash_identifier(self):
        """Test identifier hashing."""
        from app.services.encryption import encryption_service
        
        email = "test@example.com"
        hashed = encryption_service.hash_identifier(email)
        
        assert len(hashed) == 16  # Truncated hash
        assert hashed == encryption_service.hash_identifier(email)  # Consistent
        assert hashed != email  # Actually hashed
    
    def test_generate_id(self):
        """Test ID generation."""
        from app.services.encryption import encryption_service
        
        id1 = encryption_service.generate_id()
        id2 = encryption_service.generate_id()
        
        assert len(id1) > 0
        assert id1 != id2  # Unique
    
    def test_sanitize_for_logging(self):
        """Test PII sanitization."""
        from app.services.encryption import encryption_service
        
        text = "Contact john@example.com or 555-123-4567"
        sanitized = encryption_service.sanitize_for_logging(text)
        
        assert "[EMAIL]" in sanitized
        assert "[PHONE]" in sanitized
        assert "john@example.com" not in sanitized
        assert "555-123-4567" not in sanitized