"""User management API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from datetime import datetime, timedelta

from app.auth.firebase import get_current_user_uid, get_firestore_client
from app.db.models import UserStatsResponse, UsageLimits
from app.services.encryption import encryption_service
from app.logger import logger

router = APIRouter()


@router.get("/stats")
async def get_user_stats(
    user_uid: str = Depends(get_current_user_uid)
) -> UserStatsResponse:
    """Get user statistics and usage."""
    try:
        db = get_firestore_client()
        
        # Get user document
        user_ref = db.collection("users").document(user_uid)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user_data = user_doc.to_dict()
        
        # Count resumes
        resumes = user_ref.collection("resumes").stream()
        resume_count = sum(1 for _ in resumes)
        
        # Count chats
        chats = user_ref.collection("chats").stream()
        chat_count = sum(1 for _ in chats)
        
        # Get usage data
        usage = user_data.get("usage", {})
        
        return UserStatsResponse(
            total_analyses=usage.get("analyses_count", 0),
            total_tokens_used=usage.get("tokens_used", 0),
            total_resumes=resume_count,
            total_chats=chat_count,
            plan=user_data.get("plan", "free"),
            joined_date=user_data.get("created_at", datetime.utcnow())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user statistics"
        )


@router.get("/usage")
async def get_usage_limits(
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """Get user's usage limits and current usage."""
    try:
        db = get_firestore_client()
        
        # Get user document
        user_doc = db.collection("users").document(user_uid).get()
        user_data = user_doc.to_dict() if user_doc.exists else {"plan": "free"}
        
        # Get limits for plan
        limits = UsageLimits.get_limits(user_data.get("plan", "free"))
        
        # Get current usage
        usage = user_data.get("usage", {})
        
        # Count today's analyses
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        analytics = db.collection("analytics")\
            .where("user_id_hash", "==", encryption_service.hash_identifier(user_uid))\
            .where("timestamp", ">=", today_start)\
            .stream()
        
        today_analyses = sum(1 for _ in analytics)
        
        return {
            "plan": limits.plan,
            "limits": {
                "max_analyses_per_day": limits.max_analyses_per_day,
                "max_tokens_per_month": limits.max_tokens_per_month,
                "max_file_size_mb": limits.max_file_size_mb,
                "max_stored_resumes": limits.max_stored_resumes,
                "max_stored_chats": limits.max_stored_chats
            },
            "usage": {
                "analyses_today": today_analyses,
                "tokens_this_month": usage.get("tokens_used", 0),
                "stored_resumes": sum(1 for _ in db.collection("users").document(user_uid).collection("resumes").stream()),
                "stored_chats": sum(1 for _ in db.collection("users").document(user_uid).collection("chats").stream())
            },
            "remaining": {
                "analyses_today": max(0, limits.max_analyses_per_day - today_analyses),
                "tokens_this_month": max(0, limits.max_tokens_per_month - usage.get("tokens_used", 0))
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting usage limits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get usage limits"
        )


@router.delete("/account")
async def delete_account(
    confirm: bool = False,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, str]:
    """
    Delete user account and all associated data.
    
    This action is irreversible!
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please confirm account deletion by setting confirm=true"
        )
    
    try:
        db = get_firestore_client()
        user_ref = db.collection("users").document(user_uid)
        
        # Delete all subcollections
        collections = ["resumes", "chats", "memories"]
        
        for collection in collections:
            docs = user_ref.collection(collection).stream()
            for doc in docs:
                # If it's chats, delete messages too
                if collection == "chats":
                    messages = doc.reference.collection("messages").stream()
                    for msg in messages:
                        msg.reference.delete()
                doc.reference.delete()
        
        # Delete user document
        user_ref.delete()
        
        logger.info(f"Account deleted for user {user_uid[:8]}...")
        
        return {"message": "Account deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )