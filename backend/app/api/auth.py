"""Authentication API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any

from app.auth.firebase import (
    get_current_user,
    get_current_user_uid,
    get_or_create_user_doc,
    verify_firebase_token
)
from app.db.models import UserProfile, TokenResponse
from app.logger import logger

router = APIRouter()


@router.post("/verify", response_model=UserProfile)
async def verify_token(
    user: Dict[str, Any] = Depends(get_current_user)
) -> UserProfile:
    """
    Verify Firebase token and get/create user profile.
    
    This endpoint:
    1. Verifies the Firebase ID token
    2. Creates user document if first time
    3. Returns user profile
    """
    try:
        # Get or create user document
        user_doc = await get_or_create_user_doc(
            uid=user["uid"],
            email=user.get("email"),
            name=user.get("name")
        )
        
        # Create UserProfile response
        profile = UserProfile(
            uid=user["uid"],
            email=user.get("email"),
            name=user.get("name"),
            picture=user.get("picture"),
            created_at=user_doc.get("created_at"),
            plan=user_doc.get("plan", "free"),
            usage=user_doc.get("usage", {})
        )
        
        logger.info(f"User verified: {user['uid'][:8]}...")
        return profile
        
    except Exception as e:
        logger.error(f"Error verifying user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify user"
        )


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    user: Dict[str, Any] = Depends(get_current_user)
) -> UserProfile:
    """
    Get current authenticated user's profile.
    """
    try:
        # Get user document
        from app.auth.firebase import get_firestore_client
        db = get_firestore_client()
        
        user_doc = db.collection("users").document(user["uid"]).get()
        
        if not user_doc.exists:
            # Create user document if it doesn't exist
            user_data = await get_or_create_user_doc(
                uid=user["uid"],
                email=user.get("email"),
                name=user.get("name")
            )
        else:
            user_data = user_doc.to_dict()
        
        # Create UserProfile response
        profile = UserProfile(
            uid=user["uid"],
            email=user.get("email"),
            name=user.get("name"),
            picture=user.get("picture"),
            created_at=user_data.get("created_at"),
            plan=user_data.get("plan", "free"),
            usage=user_data.get("usage", {})
        )
        
        return profile
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )


@router.post("/logout")
async def logout(
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, str]:
    """
    Logout endpoint.
    
    Note: Since Firebase tokens are stateless, actual logout happens client-side.
    This endpoint can be used for logging/analytics.
    """
    logger.info(f"User logged out: {user_uid[:8]}...")
    
    # You can add additional logout logic here like:
    # - Recording logout time
    # - Clearing server-side cache
    # - Analytics
    
    return {"message": "Logged out successfully"}


@router.get("/check")
async def check_auth_status(
    token: Dict[str, Any] = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Quick auth check endpoint.
    
    Returns basic auth status without fetching full user profile.
    """
    return {
        "authenticated": True,
        "uid": token["uid"],
        "email": token.get("email"),
        "email_verified": token.get("email_verified", False)
    }