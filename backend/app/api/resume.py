"""Resume upload and management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Response
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import hashlib
from pydantic import BaseModel

from app.auth.firebase import get_current_user_uid, get_firestore_client
from app.services.encryption import encryption_service
from app.db.models import FirestoreResume, UsageLimits
from app.config.settings import settings
from app.logger import logger

router = APIRouter()


@router.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    target_role: Optional[str] = Form(None),
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Upload and store user's resume (encrypted client-side).
    
    The resume is stored once and reused across all chats.
    Frontend should encrypt the resume before sending.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed"
            )
        
        # Check file size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds {settings.max_file_size_mb}MB limit"
            )
        
        # Generate file hash for deduplication
        file_hash = hashlib.sha256(contents).hexdigest()
        
        # Get Firestore client
        db = get_firestore_client()
        
        # Check user's plan limits
        user_ref = db.collection("users").document(user_uid)
        user_doc = user_ref.get()
        user_data = user_doc.to_dict() if user_doc.exists else {"plan": "free"}
        
        limits = UsageLimits.get_limits(user_data.get("plan", "free"))
        
        # Check existing resumes count
        existing_resumes = db.collection("users").document(user_uid)\
            .collection("resumes").stream()
        resume_count = sum(1 for _ in existing_resumes)
        
        if resume_count >= limits.max_stored_resumes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Resume limit reached ({limits.max_stored_resumes} for {limits.plan} plan)"
            )
        
        # Check if this exact resume already exists
        existing = db.collection("users").document(user_uid)\
            .collection("resumes").where("file_hash", "==", file_hash).limit(1).stream()
        
        for doc in existing:
            # Resume already uploaded
            return {
                "message": "Resume already exists",
                "resume_id": doc.id,
                "is_duplicate": True
            }
        
        # Create resume document
        resume_id = encryption_service.generate_id()
        
        # Note: In production, the frontend would send already-encrypted content
        # For now, we're storing the raw content (you should encrypt this client-side)
        resume_data = FirestoreResume(
            resume_id=resume_id,
            filename=file.filename,
            encrypted_content=encryption_service.encrypt_content(contents),
            file_hash=file_hash,
            uploaded_at=datetime.now(timezone.utc),
            target_role=target_role
        )
        
        # Store in Firestore
        db.collection("users").document(user_uid)\
            .collection("resumes").document(resume_id).set(resume_data.dict())
        
        # Mark as active resume
        db.collection("users").document(user_uid).update({
            "active_resume_id": resume_id,
            "updated_at": datetime.now(timezone.utc)
        })
        
        logger.info(f"Resume uploaded for user {user_uid[:8]}... - ID: {resume_id}")
        
        return {
            "message": "Resume uploaded successfully",
            "resume_id": resume_id,
            "filename": file.filename,
            "size_mb": round(file_size_mb, 2),
            "is_duplicate": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading resume: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload resume"
        )


@router.get("/list")
async def list_resumes(
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    List all resumes for the current user.
    """
    try:
        db = get_firestore_client()
        
        # Get all resumes
        resumes_ref = db.collection("users").document(user_uid).collection("resumes")
        resumes = resumes_ref.order_by("uploaded_at", direction="DESCENDING").stream()
        
        # Get active resume ID
        user_doc = db.collection("users").document(user_uid).get()
        active_resume_id = user_doc.to_dict().get("active_resume_id") if user_doc.exists else None
        
        resume_list = []
        for doc in resumes:
            resume_data = doc.to_dict()
            resume_list.append({
                "resume_id": doc.id,
                "filename": resume_data.get("filename"),
                "uploaded_at": resume_data.get("uploaded_at"),
                "analysis_count": resume_data.get("analysis_count", 0),
                "target_role": resume_data.get("target_role", ""),  # Add this line
                "is_active": doc.id == active_resume_id
            })
        
        return {
            "resumes": resume_list,
            "count": len(resume_list),
            "active_resume_id": active_resume_id
        }
        
    except Exception as e:
        logger.error(f"Error listing resumes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list resumes"
        )


@router.post("/{resume_id}/set-active")
async def set_active_resume(
    resume_id: str,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, str]:
    """
    Set a resume as the active one for analysis.
    """
    try:
        db = get_firestore_client()
        
        # Verify resume exists
        resume_ref = db.collection("users").document(user_uid)\
            .collection("resumes").document(resume_id)
        
        if not resume_ref.get().exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        # Update active resume
        db.collection("users").document(user_uid).update({
            "active_resume_id": resume_id,
            "updated_at": datetime.now(timezone.utc)
        })
        
        logger.info(f"Active resume set for user {user_uid[:8]}... - ID: {resume_id}")
        
        return {"message": "Active resume updated", "resume_id": resume_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting active resume: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set active resume"
        )

@router.get("/{resume_id}/download")
async def download_resume(
    resume_id: str,
    user_uid: str = Depends(get_current_user_uid)
) -> Response:
    """Download a resume file."""
    try:
        db = get_firestore_client()
        
        resume_ref = db.collection("users").document(user_uid)\
            .collection("resumes").document(resume_id)
        
        resume_doc = resume_ref.get()
        if not resume_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        resume_data = resume_doc.to_dict()
        
        # Convert hex string back to bytes
        content = encryption_service.decrypt_content(resume_data.get("encrypted_content", ""))
        
        return Response(
            content=content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={resume_data.get('filename', 'resume.pdf')}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading resume: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download resume"
        )

@router.delete("/{resume_id}")
async def delete_resume(
    resume_id: str,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, str]:
    """
    Delete a resume.
    """
    try:
        db = get_firestore_client()
        
        # Delete resume document
        resume_ref = db.collection("users").document(user_uid)\
            .collection("resumes").document(resume_id)
        
        if not resume_ref.get().exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        resume_ref.delete()
        
        # Check if this was the active resume
        user_doc = db.collection("users").document(user_uid).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            if user_data.get("active_resume_id") == resume_id:
                # Clear active resume
                db.collection("users").document(user_uid).update({
                    "active_resume_id": None,
                    "updated_at": datetime.now(timezone.utc)
                })
        
        logger.info(f"Resume deleted for user {user_uid[:8]}... - ID: {resume_id}")
        
        return {"message": "Resume deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting resume: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete resume"
        )

class UpdateTargetRoleRequest(BaseModel):
    target_role: str

@router.patch("/{resume_id}/target-role")
async def update_target_role(
    resume_id: str,
    request: UpdateTargetRoleRequest,  # Changed from target_role: str
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, str]:
    """Update the target role for a resume."""
    try:
        db = get_firestore_client()
        
        # Verify resume exists
        resume_ref = db.collection("users").document(user_uid)\
            .collection("resumes").document(resume_id)
        
        if not resume_ref.get().exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        # Update target role
        resume_ref.update({
            "target_role": request.target_role,
            "updated_at": datetime.now(timezone.utc)
        })
        
        logger.info(f"Target role updated for resume {resume_id[:8]}...")
        
        return {"message": "Target role updated", "target_role": request.target_role}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating target role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update target role"
        )