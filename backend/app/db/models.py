"""Database models and schemas for AppSageAI."""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# Enums
class AnalysisType(str, Enum):
    """Types of analysis available."""
    RESUME_REVIEW = "resume_review"
    SKILL_IMPROVEMENT = "skill_improvement"
    KEYWORD_ANALYSIS = "keyword_analysis"
    PERCENTAGE_MATCH = "percentage_match"
    COVER_LETTER = "cover_letter"
    CUSTOM_QUERY = "custom_query"


class FeedbackType(str, Enum):
    """Types of feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


# Request Models
class ResumeAnalysisRequest(BaseModel):
    """Request model for resume analysis."""
    job_description: str = Field(..., min_length=1, max_length=10000)
    analysis_type: AnalysisType
    custom_query: Optional[str] = Field(None, max_length=1000)
    encrypted_resume: str = Field(..., description="Client-encrypted resume content")
    
    @field_validator('custom_query')
    def validate_custom_query(cls, v, values):
        if values.get('analysis_type') == AnalysisType.CUSTOM_QUERY and not v:
            raise ValueError("Custom query is required for custom analysis type")
        return v


class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    encrypted_content: str = Field(..., description="Encrypted message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    """Request model for feedback."""
    chat_id: str = Field(..., min_length=1)
    message_id: str = Field(..., min_length=1)
    feedback_type: FeedbackType
    comment: Optional[str] = Field(None, max_length=1000)


# Response Models
class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    analysis_id: str
    encrypted_response: str = Field(..., description="Encrypted analysis result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Non-sensitive metadata (tokens, time, etc.)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(BaseModel):
    """Model for chat session."""
    session_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0


class UserProfile(BaseModel):
    """User profile model."""
    uid: str
    email: Optional[str] = None
    name: Optional[str] = None
    picture: Optional[str] = None
    created_at: datetime
    plan: str = "free"
    usage: Dict[str, Any] = Field(default_factory=dict)


# Firestore Document Models
class FirestoreChat(BaseModel):
    """Firestore chat document."""
    session_id: str
    encrypted_messages: List[Dict[str, Any]]
    job_description_hash: str  # Hash of JD for grouping
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FirestoreMemory(BaseModel):
    """Firestore memory document."""
    memory_id: str
    memory_type: str  # skills, experience, preferences
    encrypted_content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FirestoreResume(BaseModel):
    """Firestore resume document."""
    resume_id: str
    filename: str
    encrypted_content: str
    file_hash: str  # Hash to detect duplicates
    uploaded_at: datetime
    analysis_count: int = 0
    last_analyzed: Optional[datetime] = None
    target_role: Optional[str] = None  # ADD THIS LINE
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FirestoreAnalytics(BaseModel):
    """Analytics document (backend only, no encryption needed)."""
    user_id_hash: str  # Hashed user ID for privacy
    action: str
    analysis_type: Optional[AnalysisType] = None
    timestamp: datetime
    response_time_ms: int
    tokens_used: int
    model_used: str
    success: bool = True
    error: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# API Response Models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    app: str
    version: str
    environment: str


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    request_id: Optional[str] = None
    type: Optional[str] = None


class TokenResponse(BaseModel):
    """Token response for authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class UserStatsResponse(BaseModel):
    """User statistics response."""
    total_analyses: int
    total_tokens_used: int
    total_resumes: int
    total_chats: int
    plan: str
    joined_date: datetime


class ChatHistoryResponse(BaseModel):
    """Chat history response."""
    chats: List[ChatSession]
    total: int
    page: int
    page_size: int
    has_next: bool

class TrackerStatus(str, Enum):
    """Job application tracking status."""
    NOT_APPLICABLE = "not_applicable"
    INTERESTED = "interested"
    APPLIED = "applied"
    INTERVIEWING = "interviewing"
    OFFERED = "offered"
    REJECTED = "rejected"

# Limits for free tier
class UsageLimits(BaseModel):
    """Usage limits for different plans."""
    plan: str
    max_analyses_per_day: int = Field(default=10)
    max_tokens_per_month: int = Field(default=100000)
    max_file_size_mb: int = Field(default=10)
    max_stored_resumes: int = Field(default=5)
    max_stored_chats: int = Field(default=50)
    
    @classmethod
    def get_limits(cls, plan: str) -> "UsageLimits":
        """Get limits for a specific plan."""
        limits_config = {
            "free": {
                "max_analyses_per_day": 10,
                "max_tokens_per_month": 100000,
                "max_file_size_mb": 10,
                "max_stored_resumes": 5,
                "max_stored_chats": 50
            },
            "pro": {
                "max_analyses_per_day": 100,
                "max_tokens_per_month": 1000000,
                "max_file_size_mb": 25,
                "max_stored_resumes": 50,
                "max_stored_chats": 500
            }
        }
        
        config = limits_config.get(plan, limits_config["free"])
        return cls(plan=plan, **config)