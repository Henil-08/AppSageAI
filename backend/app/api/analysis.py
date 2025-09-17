"""Analysis API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

from app.auth.firebase import get_current_user_uid, get_firestore_client
from app.services.encryption import encryption_service
from app.core.resume_analyzer import ResumeAnalyzer
from app.core.prompts import PromptManager
from app.db.models import AnalysisType, FirestoreAnalytics
from app.config.settings import settings
from app.logger import logger

router = APIRouter()

# Request models
class AnalysisRequest(BaseModel):
    analysis_type: str
    custom_query: Optional[str] = None
    resume_id: Optional[str] = None

class ExtractJobDetailsRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    feedback_type: str
    comment: Optional[str] = None

# Initialize analyzer and prompt manager
analyzer = ResumeAnalyzer()
prompt_manager = PromptManager()

@router.post("/analyze/{session_id}")
async def analyze_chat(
    session_id: str,
    request: AnalysisRequest,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Perform analysis on a chat session.
    """
    try:
        db = get_firestore_client()
        
        # Get user document FIRST - FIX THE ERROR
        user_ref = db.collection("users").document(user_uid)
        user_doc = user_ref.get()
        
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
        
        # Get active resume or specified resume
        resume_id = request.resume_id
        if not resume_id:
            if user_doc.exists:
                resume_id = user_doc.to_dict().get("active_resume_id")
        
        if not resume_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No resume selected. Please upload a resume first."
            )
        
        # Get resume content
        resume_ref = db.collection("users").document(user_uid)\
            .collection("resumes").document(resume_id)
        
        resume_doc = resume_ref.get()
        if not resume_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )
        
        resume_data = resume_doc.to_dict()
        resume_content = bytes.fromhex(resume_data.get("encrypted_content", ""))
        
        # Get job description from chat
        job_description = chat_data.get("job_description", "")
        
        # Get appropriate prompt
        analysis_type = AnalysisType(request.analysis_type)
        prompt_template = prompt_manager.get_prompt(
            analysis_type,
            custom_query=request.custom_query
        )
        
        # Perform analysis
        result, metadata = await analyzer.analyze(
            resume_content=resume_content,
            job_description=job_description,
            analysis_type=analysis_type,
            prompt_template=prompt_template,
            user_name=user_doc.to_dict().get("name", "Candidate") if user_doc.exists else "Candidate",
            custom_query=request.custom_query
        )
        
        # Generate analysis ID
        analysis_id = encryption_service.generate_id()
        
        # Store analysis result as a message in the chat
        message_data = {
            "message_id": analysis_id,
            "role": "assistant",
            "encrypted_content": result,  # In production, encrypt this
            "timestamp": datetime.utcnow(),
            "metadata": {
                "analysis_type": request.analysis_type,
                "tokens_used": metadata.get("tokens_used", 0),
                "response_time_ms": metadata.get("response_time_ms", 0),
                "model": metadata.get("model", settings.model_name)
            }
        }
        
        # Add message to chat
        chat_ref.collection("messages").document(analysis_id).set(message_data)
        
        # Update chat metadata
        chat_ref.update({
            "updated_at": datetime.utcnow(),
            "message_count": chat_data.get("message_count", 0) + 1,
            "metadata.last_analysis_type": request.analysis_type,
            "metadata.total_tokens_used": chat_data.get("metadata", {}).get("total_tokens_used", 0) + metadata.get("tokens_used", 0)
        })
        
        # Log analytics (without PII)
        analytics_data = FirestoreAnalytics(
            user_id_hash=encryption_service.hash_identifier(user_uid),
            action="analysis_performed",
            analysis_type=analysis_type,
            timestamp=datetime.utcnow(),
            response_time_ms=metadata.get("response_time_ms", 0),
            tokens_used=metadata.get("tokens_used", 0),
            model_used=metadata.get("model", settings.model_name),
            success=True
        )
        
        db.collection("analytics").add(analytics_data.dict())
        
        logger.info(f"Analysis completed for session {session_id[:8]}... Type: {request.analysis_type}")
        
        return {
            "analysis_id": analysis_id,
            "encrypted_response": result,  # In production, this should be encrypted
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing analysis: {e}")
        
        # Log failed analytics
        analytics_data = FirestoreAnalytics(
            user_id_hash=encryption_service.hash_identifier(user_uid),
            action="analysis_failed",
            analysis_type=AnalysisType(request.analysis_type) if request.analysis_type else None,
            timestamp=datetime.utcnow(),
            response_time_ms=0,
            tokens_used=0,
            model_used=settings.model_name,
            success=False,
            error=str(e)
        )
        
        db.collection("analytics").add(analytics_data.dict())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform analysis"
        )

@router.post("/extract-job-details")
async def extract_job_details_endpoint(
    request: ExtractJobDetailsRequest,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Extract job details from text using LLM.
    """
    try:
        from app.core.job_extractor import extract_job_details
        
        # Use Groq to extract details
        details = await extract_job_details(request.text)
        
        return {
            "job_title": details.get("job_title", ""),
            "company": details.get("company", ""),
            "job_description": details.get("job_description", request.text),
            "location": details.get("location", ""),
            "salary": details.get("salary", ""),
            "job_type": details.get("job_type", ""),
            "is_job_listing": details.get("is_job_listing", False)
        }
        
    except Exception as e:
        logger.error(f"Error extracting job details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract job details"
        )

@router.get("/quick-actions/{session_id}")
async def get_quick_actions(
    session_id: str,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Get available quick actions for a chat session.
    """
    return {
        "session_id": session_id,
        "actions": [
            {
                "id": "job_match",
                "type": "resume_review",
                "label": "📊 Job Match Analysis",
                "description": "Comprehensive review against job requirements",
                "icon": "📊"
            },
            {
                "id": "ats_scan",
                "type": "keyword_analysis",
                "label": "🎯 ATS Scan",
                "description": "Check ATS compatibility and keywords",
                "icon": "🎯"
            },
            {
                "id": "match_percentage",
                "type": "percentage_match",
                "label": "📈 Match Percentage",
                "description": "Calculate your compatibility score",
                "icon": "📈"
            },
            {
                "id": "skill_gap",
                "type": "skill_improvement",
                "label": "🚀 Skill Improvement",
                "description": "Identify gaps and improvement areas",
                "icon": "🚀"
            },
            {
                "id": "cover_letter",
                "type": "cover_letter",
                "label": "✉️ Cover Letter",
                "description": "Generate a tailored cover letter",
                "icon": "✉️"
            }
        ]
    }

@router.post("/feedback/{analysis_id}")
async def submit_feedback(
    analysis_id: str,
    request: FeedbackRequest,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, str]:
    """
    Submit feedback for an analysis.
    """
    try:
        db = get_firestore_client()
        
        # Store feedback
        feedback_data = {
            "analysis_id": analysis_id,
            "user_id_hash": encryption_service.hash_identifier(user_uid),
            "feedback_type": request.feedback_type,
            "comment": request.comment,
            "timestamp": datetime.utcnow()
        }
        
        db.collection("feedback").add(feedback_data)
        
        logger.info(f"Feedback submitted for analysis {analysis_id[:8]}...")
        
        return {"message": "Thank you for your feedback!"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )