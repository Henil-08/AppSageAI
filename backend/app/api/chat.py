"""Chat session management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
from pydantic import BaseModel

from app.auth.firebase import get_current_user_uid, get_firestore_client
from app.services.encryption import encryption_service
from app.db.models import ChatMessage, ChatSession, FirestoreChat
from app.config.settings import settings
from app.logger import logger

class UpdateChatRequest(BaseModel):
    job_title: str
    company: str
    job_description: str

router = APIRouter()


@router.post("/create")
async def create_chat_session(
    job_description: Optional[str] = None,
    job_title: Optional[str] = None,
    company: Optional[str] = None,
    initial_message: Optional[str] = None,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Create a new chat session.
    """
    try:
        db = get_firestore_client()
        
        # Check if user has an active resume
        user_doc = db.collection("users").document(user_uid).get()
        if not user_doc.exists or not user_doc.to_dict().get("active_resume_id"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please upload a resume first"
            )
        
        active_resume_id = user_doc.to_dict()["active_resume_id"]
        
        # Generate session ID
        session_id = encryption_service.generate_id()
        
        # Create hash of job description for grouping similar jobs
        jd_hash = hashlib.sha256((job_description or "general").encode()).hexdigest()[:16]
        
        # Create chat session
        chat_data = {
            "session_id": session_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "job_description": job_description or "General consultation",
            "job_description_hash": jd_hash,
            "job_title": job_title or "General Consultation",
            "company": company or "Career Development",
            "resume_id": active_resume_id,
            "message_count": 0,
            "metadata": {
                "last_analysis_type": None,
                "total_tokens_used": 0
            }
        }
        
        # Store in Firestore
        db.collection("users").document(user_uid)\
            .collection("chats").document(session_id).set(chat_data)
        
        logger.info(f"Chat session created for user {user_uid[:8]}... - ID: {session_id}")
        
        return {
            "session_id": session_id,
            "message": "Chat session created",
            "job_title": job_title or "General Consultation",
            "company": company or "Career Development",
            "resume_id": active_resume_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat session"
        )


@router.get("/list")
async def list_chat_sessions(
    limit: int = 20,
    offset: int = 0,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    List all chat sessions for the user.
    """
    try:
        db = get_firestore_client()
        
        # Get chat sessions
        chats_ref = db.collection("users").document(user_uid).collection("chats")
        chats_query = chats_ref.order_by("updated_at", direction="DESCENDING")\
            .limit(limit).offset(offset)
        
        chats = chats_query.stream()
        
        chat_list = []
        for doc in chats:
            chat_data = doc.to_dict()
            chat_list.append({
                "session_id": doc.id,
                "created_at": chat_data.get("created_at"),
                "updated_at": chat_data.get("updated_at"),
                "job_title": chat_data.get("job_title", "Untitled"),
                "company": chat_data.get("company", "Unknown"),
                "message_count": chat_data.get("message_count", 0),
                "preview": chat_data.get("job_description", "")[:100] + "...",
                "tracker_status": chat_data.get("tracker_status", "not_applicable"),
                "applied_date": chat_data.get("applied_date"),
                "tracker_notes": chat_data.get("tracker_notes")
            })
        
        # Get total count
        total_count = sum(1 for _ in chats_ref.stream())
        
        return {
            "chats": chat_list,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }
        
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list chat sessions"
        )


@router.get("/{session_id}")
async def get_chat_session(
    session_id: str,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Get a specific chat session with all messages.
    """
    try:
        db = get_firestore_client()
        
        # Get chat session
        chat_ref = db.collection("users").document(user_uid)\
            .collection("chats").document(session_id)
        
        chat_doc = chat_ref.get()
        if not chat_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        chat_data = chat_doc.to_dict()
        
        # Get messages (they're encrypted, so we just pass them through)
        messages_ref = chat_ref.collection("messages").order_by("timestamp")
        messages = []
        
        for msg_doc in messages_ref.stream():
            msg_data = msg_doc.to_dict()
            messages.append({
                "message_id": msg_doc.id,
                "role": msg_data.get("role"),
                "encrypted_content": msg_data.get("encrypted_content"),
                "timestamp": msg_data.get("timestamp"),
                "metadata": msg_data.get("metadata", {})
            })
        
        return {
            "session_id": session_id,
            "created_at": chat_data.get("created_at"),
            "job_title": chat_data.get("job_title"),
            "company": chat_data.get("company"),
            "job_description": chat_data.get("job_description"),
            "resume_id": chat_data.get("resume_id"),
            "messages": messages,
            "message_count": len(messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chat session"
        )


@router.post("/{session_id}/message")
async def add_message(
    session_id: str,
    message: ChatMessage,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Add a message to the chat session.
    Messages should be encrypted client-side.
    """
    try:
        db = get_firestore_client()
        
        # Verify chat session exists
        chat_ref = db.collection("users").document(user_uid)\
            .collection("chats").document(session_id)
        
        if not chat_ref.get().exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        # Generate message ID
        message_id = encryption_service.generate_id()
        
        # Store message
        message_data = {
            "role": message.role,
            "encrypted_content": message.encrypted_content,
            "timestamp": datetime.utcnow(),
            "metadata": message.metadata or {}
        }
        
        chat_ref.collection("messages").document(message_id).set(message_data)
        
        # Update chat session
        chat_ref.update({
            "updated_at": datetime.utcnow(),
            "message_count": chat_ref.get().to_dict().get("message_count", 0) + 1
        })
        
        logger.info(f"Message added to chat {session_id[:8]}... - ID: {message_id}")
        
        return {
            "message_id": message_id,
            "timestamp": message_data["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add message"
        )

@router.patch("/{session_id}/update")
async def update_chat_details(
    session_id: str,
    request: UpdateChatRequest,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Update chat session details (title, company, job description).
    """
    try:
        db = get_firestore_client()
        
        # Get chat reference
        chat_ref = db.collection("users").document(user_uid)\
            .collection("chats").document(session_id)
        
        # Check if chat exists
        chat_doc = chat_ref.get()
        if not chat_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        # Update the chat document
        update_data = {
            "job_title": request.job_title,
            "company": request.company,
            "job_description": request.job_description,
            "updated_at": datetime.utcnow()
        }
        
        chat_ref.update(update_data)
        
        logger.info(f"Chat {session_id[:8]}... updated - Title: {request.job_title}")
        
        return {
            "message": "Chat details updated successfully",
            "session_id": session_id,
            "job_title": request.job_title,
            "company": request.company
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chat details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update chat details"
        )
    
@router.patch("/{session_id}/tracker")
async def update_tracker_status(
    session_id: str,
    tracker_status: str,
    applied_date: Optional[str] = None,
    notes: Optional[str] = None,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """Update job tracker status for a chat."""
    try:
        db = get_firestore_client()
        
        chat_ref = db.collection("users").document(user_uid)\
            .collection("chats").document(session_id)
        
        if not chat_ref.get().exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        update_data = {
            "tracker_status": tracker_status,
            "tracker_updated_at": datetime.utcnow()
        }
        
        if applied_date:
            update_data["applied_date"] = applied_date
        if notes:
            update_data["tracker_notes"] = notes
            
        chat_ref.update(update_data)
        
        return {"message": "Tracker status updated", "status": tracker_status}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating tracker status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update tracker status"
        )

@router.delete("/{session_id}")
async def delete_chat_session(
    session_id: str,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, str]:
    """
    Delete a chat session and all its messages.
    """
    try:
        db = get_firestore_client()
        
        # Get chat reference
        chat_ref = db.collection("users").document(user_uid)\
            .collection("chats").document(session_id)
        
        if not chat_ref.get().exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        # Delete all messages first
        messages = chat_ref.collection("messages").stream()
        for msg in messages:
            msg.reference.delete()
        
        # Delete chat document
        chat_ref.delete()
        
        logger.info(f"Chat session deleted: {session_id[:8]}...")
        
        return {"message": "Chat session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat session"
        )