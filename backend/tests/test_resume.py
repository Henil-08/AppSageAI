"""Tests for resume management endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from io import BytesIO


class TestResumeEndpoints:
    """Test resume management endpoints."""
    
    def test_upload_resume_success(
        self,
        test_client,
        mock_verify_token,
        mock_firestore_client,
        mock_encryption_service
    ):
        """Test successful resume upload."""
        # Mock Firestore responses
        mock_user_doc = MagicMock()
        mock_user_doc.exists = True
        mock_user_doc.to_dict.return_value = {"plan": "free"}
        
        mock_firestore_client.collection().document().get.return_value = mock_user_doc
        mock_firestore_client.collection().document().collection().stream.return_value = []
        mock_firestore_client.collection().document().collection().where().limit().stream.return_value = []
        
        # Create test PDF file
        pdf_content = b"%PDF-1.4\n%Test PDF content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
        
        # Make request
        response = test_client.post(
            "/api/v1/resume/upload",
            files=files,
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Resume uploaded successfully"
        assert data["resume_id"] == "test_id_123"
        assert data["filename"] == "test.pdf"
        assert data["is_duplicate"] == False
    
    def test_upload_resume_invalid_type(
        self,
        test_client,
        mock_verify_token,
        mock_firestore_client
    ):
        """Test resume upload with invalid file type."""
        # Create test non-PDF file
        files = {"file": ("test.txt", BytesIO(b"text content"), "text/plain")}
        
        # Make request
        response = test_client.post(
            "/api/v1/resume/upload",
            files=files,
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 400
        assert "Only PDF files are allowed" in response.json()["detail"]
    
    def test_upload_resume_too_large(
        self,
        test_client,
        mock_verify_token,
        mock_firestore_client
    ):
        """Test resume upload with file too large."""
        # Create large fake PDF (11MB)
        large_content = b"%PDF-1.4\n" + b"x" * (11 * 1024 * 1024)
        files = {"file": ("large.pdf", BytesIO(large_content), "application/pdf")}
        
        # Make request
        response = test_client.post(
            "/api/v1/resume/upload",
            files=files,
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 400
        assert "File size exceeds" in response.json()["detail"]
    
    def test_upload_resume_duplicate(
        self,
        test_client,
        mock_verify_token,
        mock_firestore_client,
        mock_encryption_service
    ):
        """Test uploading duplicate resume."""
        # Mock Firestore responses
        mock_user_doc = MagicMock()
        mock_user_doc.exists = True
        mock_user_doc.to_dict.return_value = {"plan": "free"}
        
        # Mock existing resume with same hash
        mock_existing_doc = MagicMock()
        mock_existing_doc.id = "existing_resume_id"
        
        mock_firestore_client.collection().document().get.return_value = mock_user_doc
        mock_firestore_client.collection().document().collection().stream.return_value = []
        mock_firestore_client.collection().document().collection().where().limit().stream.return_value = [mock_existing_doc]
        
        # Create test PDF file
        pdf_content = b"%PDF-1.4\n%Test PDF content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
        
        # Make request
        response = test_client.post(
            "/api/v1/resume/upload",
            files=files,
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Resume already exists"
        assert data["resume_id"] == "existing_resume_id"
        assert data["is_duplicate"] == True
    
    def test_list_resumes(
        self,
        test_client,
        mock_verify_token,
        mock_firestore_client
    ):
        """Test listing user's resumes."""
        # Mock user document
        mock_user_doc = MagicMock()
        mock_user_doc.exists = True
        mock_user_doc.to_dict.return_value = {"active_resume_id": "res_001"}
        
        # Mock resume documents
        mock_resume_1 = MagicMock()
        mock_resume_1.id = "res_001"
        mock_resume_1.to_dict.return_value = {
            "filename": "resume1.pdf",
            "uploaded_at": "2024-01-15T10:00:00",
            "analysis_count": 5
        }
        
        mock_resume_2 = MagicMock()
        mock_resume_2.id = "res_002"
        mock_resume_2.to_dict.return_value = {
            "filename": "resume2.pdf",
            "uploaded_at": "2024-01-14T10:00:00",
            "analysis_count": 2
        }
        
        mock_firestore_client.collection().document().get.return_value = mock_user_doc
        mock_firestore_client.collection().document().collection().order_by().stream.return_value = [
            mock_resume_1, mock_resume_2
        ]
        
        # Make request
        response = test_client.get(
            "/api/v1/resume/list",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["active_resume_id"] == "res_001"
        assert len(data["resumes"]) == 2
        assert data["resumes"][0]["is_active"] == True
        assert data["resumes"][1]["is_active"] == False
    
    def test_set_active_resume(
        self,
        test_client,
        mock_verify_token,
        mock_firestore_client
    ):
        """Test setting active resume."""
        # Mock resume exists
        mock_resume_doc = MagicMock()
        mock_resume_doc.exists = True
        
        mock_firestore_client.collection().document().collection().document().get.return_value = mock_resume_doc
        
        # Make request
        response = test_client.post(
            "/api/v1/resume/res_002/set-active",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Active resume updated"
        assert data["resume_id"] == "res_002"
        
        # Verify update was called
        mock_firestore_client.collection().document().update.assert_called_once()
    
    def test_set_active_resume_not_found(
        self,
        test_client,
        mock_verify_token,
        mock_firestore_client
    ):
        """Test setting active resume that doesn't exist."""
        # Mock resume doesn't exist
        mock_resume_doc = MagicMock()
        mock_resume_doc.exists = False
        
        mock_firestore_client.collection().document().collection().document().get.return_value = mock_resume_doc
        
        # Make request
        response = test_client.post(
            "/api/v1/resume/nonexistent/set-active",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 404
        assert "Resume not found" in response.json()["detail"]
    
    def test_delete_resume(
        self,
        test_client,
        mock_verify_token,
        mock_firestore_client
    ):
        """Test deleting a resume."""
        # Mock resume exists
        mock_resume_doc = MagicMock()
        mock_resume_doc.exists = True
        
        # Mock user doc
        mock_user_doc = MagicMock()
        mock_user_doc.exists = True
        mock_user_doc.to_dict.return_value = {"active_resume_id": "res_001"}
        
        mock_firestore_client.collection().document().collection().document().get.return_value = mock_resume_doc
        mock_firestore_client.collection().document().get.return_value = mock_user_doc
        
        # Make request
        response = test_client.delete(
            "/api/v1/resume/res_001",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        
        # Verify delete was called
        mock_firestore_client.collection().document().collection().document().delete.assert_called_once()
        # Verify active resume was cleared
        mock_firestore_client.collection().document().update.assert_called_once()