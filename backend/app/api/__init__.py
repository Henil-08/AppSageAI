"""API router initialization."""

from fastapi import APIRouter

from app.api.auth import router as auth_router
from app.api.analysis import router as analysis_router
from app.api.chat import router as chat_router
from app.api.user import router as user_router
from app.api.resume import router as resume_router
from app.api.prompts import router as prompts_router

# Create main API router
router = APIRouter()

# Include sub-routers
router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
router.include_router(resume_router, prefix="/resume", tags=["Resume"])
router.include_router(chat_router, prefix="/chat", tags=["Chat"])
router.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])
router.include_router(user_router, prefix="/user", tags=["User"])
router.include_router(prompts_router, prefix="/prompts", tags=["Prompts"])

# API root endpoint
@router.get("/")
async def api_root():
    """API root endpoint."""
    return {
        "message": "AppSageAI API v1",
        "endpoints": {
            "auth": "/api/v1/auth",
            "analysis": "/api/v1/analysis",
            "chat": "/api/v1/chat",
            "user": "/api/v1/user"
        }
    }