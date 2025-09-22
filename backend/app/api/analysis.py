"""Analysis API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel

from app.auth.firebase import get_current_user_uid, get_firestore_client
from app.services.encryption import encryption_service
from app.core.resume_analyzer import ResumeAnalyzer
from app.core.prompts import PromptManager
from app.core.job_extractor import extract_job_details
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

@router.post("/analyze-stream/{session_id}")
async def analyze_chat_stream(
    session_id: str,
    request: AnalysisRequest,
    user_uid: str = Depends(get_current_user_uid)
):
    """
    Perform analysis on a chat session with streaming response.
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
        
        encrypted_content = resume_data.get("encrypted_content", "")
        # Try to decode from hex first (old format), if that fails, decrypt directly
        try:
            resume_content = bytes.fromhex(encrypted_content)
        except ValueError:
            # It's already encrypted bytes or base64, decrypt it
            resume_content = encryption_service.decrypt_content(encrypted_content)
        
        # Get job description from chat
        job_description = chat_data.get("job_description", "")
        
        # Avoid parsing Initial Job Description.
        if str(job_description).lower() == "general consultation" or str(job_description).lower() == "career development":
            job_description = ""

        # Get appropriate prompt
        analysis_type = AnalysisType(request.analysis_type)

        # Check if user has custom prompt for this analysis type
        custom_prompt = None
        try:
            prompts_ref = db.collection("users").document(user_uid)\
                .collection("custom_prompts").document(request.analysis_type)
            prompt_doc = prompts_ref.get()
            if prompt_doc.exists:
                encrypted_prompt = prompt_doc.to_dict().get("prompt_template")

                # Decrypt the custom prompt
                try:
                    custom_prompt = encryption_service.decrypt_content(encrypted_prompt).decode()
                except:
                    # If decryption fails, might be old unencrypted data
                    custom_prompt = encrypted_prompt
                
                # IMPORTANT: Ensure {context} is in the prompt for RAG to work
                if "{context}" not in custom_prompt:
                    # If user removed {context}, add it at the end
                    custom_prompt += "\n\nContext from resume:\n{context}"  
        except Exception as e:
            logger.warning(f"Error loading custom prompt: {e}")
            pass  # Use default if error

        # Use custom prompt if available, otherwise use default
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = prompt_manager.get_prompt(
                analysis_type,
                custom_query=request.custom_query
            )

        # Create a generator function for streaming
        async def generate():
            try:
                # Get all the same data as before
                user_ref = db.collection("users").document(user_uid)
                user_doc = user_ref.get()
                
                # ... [All the same setup code up to getting the prompt_template]
                
                # Initialize the analyzer for streaming
                result_chunks = []
                metadata = {}
                
                # Call the streaming version of analyze
                async for chunk in analyzer.analyze_stream(
                    resume_content=resume_content,
                    job_description=job_description,
                    analysis_type=analysis_type,
                    prompt_template=prompt_template,
                    user_name=user_doc.to_dict().get("name", "Candidate") if user_doc.exists else "Candidate",
                    custom_query=request.custom_query
                ):
                    if chunk.get("type") == "token":
                        # Stream the token
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk['content']})}\n\n"
                        result_chunks.append(chunk['content'])
                    elif chunk.get("type") == "metadata":
                        metadata = chunk['metadata']
                        yield f"data: {json.dumps({'type': 'metadata', 'metadata': metadata})}\n\n"
                
                # After streaming is complete, save to database
                full_result = ''.join(result_chunks)
                analysis_id = encryption_service.generate_id()
                
                # Store the complete message
                message_data = {
                    "message_id": analysis_id,
                    "role": "assistant",
                    "encrypted_content": encryption_service.encrypt_content(full_result.encode()),
                    "timestamp": datetime.now(timezone.utc),
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
                    "updated_at": datetime.now(timezone.utc),
                    "message_count": chat_doc.to_dict().get("message_count", 0) + 1,
                })
                
                # Send completion signal with analysis_id
                yield f"data: {json.dumps({'type': 'done', 'analysis_id': analysis_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable Nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform streaming analysis"
        )
        

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
        
        encrypted_content = resume_data.get("encrypted_content", "")
        # Try to decode from hex first (old format), if that fails, decrypt directly
        try:
            resume_content = bytes.fromhex(encrypted_content)
        except ValueError:
            # It's already encrypted bytes or base64, decrypt it
            resume_content = encryption_service.decrypt_content(encrypted_content)
        
        # Get job description from chat
        job_description = chat_data.get("job_description", "")
        
        # Avoid parsing Initial Job Description.
        if str(job_description).lower() == "general consultation" or str(job_description).lower() == "career development":
            job_description = ""

        # Get appropriate prompt
        analysis_type = AnalysisType(request.analysis_type)

        # Check if user has custom prompt for this analysis type
        custom_prompt = None
        try:
            prompts_ref = db.collection("users").document(user_uid)\
                .collection("custom_prompts").document(request.analysis_type)
            prompt_doc = prompts_ref.get()
            if prompt_doc.exists:
                encrypted_prompt = prompt_doc.to_dict().get("prompt_template")

                # Decrypt the custom prompt
                try:
                    custom_prompt = encryption_service.decrypt_content(encrypted_prompt).decode()
                except:
                    # If decryption fails, might be old unencrypted data
                    custom_prompt = encrypted_prompt
                
                # IMPORTANT: Ensure {context} is in the prompt for RAG to work
                if "{context}" not in custom_prompt:
                    # If user removed {context}, add it at the end
                    custom_prompt += "\n\nContext from resume:\n{context}"  
        except Exception as e:
            logger.warning(f"Error loading custom prompt: {e}")
            pass  # Use default if error

        # Use custom prompt if available, otherwise use default
        if custom_prompt:
            prompt_template = custom_prompt
        else:
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
            "encrypted_content": encryption_service.encrypt_content(result.encode()),
            "timestamp": datetime.now(timezone.utc),
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
            "updated_at": datetime.now(timezone.utc),
            "message_count": chat_data.get("message_count", 0) + 1,
            "metadata.last_analysis_type": request.analysis_type,
            "metadata.total_tokens_used": chat_data.get("metadata", {}).get("total_tokens_used", 0) + metadata.get("tokens_used", 0)
        })
        
        # Log analytics (without PII)
        analytics_data = FirestoreAnalytics(
            user_id_hash=encryption_service.hash_identifier(user_uid),
            action="analysis_performed",
            analysis_type=analysis_type,
            timestamp=datetime.now(timezone.utc),
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
            "timestamp": datetime.now(timezone.utc).isoformat()
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
            timestamp=datetime.now(timezone.utc),
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
        
        # Use Gemini to extract details
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

@router.post("/feedback/{message_id}")
async def submit_feedback(
    message_id: str,
    request: FeedbackRequest,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Submit or toggle feedback for a message.
    If feedback exists and same type, remove it (toggle off).
    If feedback exists and different type, update it.
    If no feedback exists, create it.
    """
    try:
        db = get_firestore_client()
        
        # Create a unique feedback document ID based on user and message
        feedback_doc_id = f"{user_uid}_{message_id}"
        feedback_ref = db.collection("feedback").document(feedback_doc_id)
        
        existing_feedback = feedback_ref.get()
        
        if existing_feedback.exists:
            existing_data = existing_feedback.to_dict()
            
            # If same feedback type, remove it (toggle off)
            if existing_data.get("feedback_type") == request.feedback_type:
                feedback_ref.delete()
                logger.info(f"Feedback removed for message {message_id[:8]}...")
                return {
                    "message": "Feedback removed",
                    "feedback_type": None
                }
            else:
                # Different feedback type, update it
                feedback_ref.update({
                    "feedback_type": request.feedback_type,
                    "updated_at": datetime.now(timezone.utc)
                })
                logger.info(f"Feedback updated for message {message_id[:8]}...")
                return {
                    "message": "Feedback updated",
                    "feedback_type": request.feedback_type
                }
        else:
            # No existing feedback, create new
            feedback_data = {
                "message_id": message_id,
                "user_id_hash": encryption_service.hash_identifier(user_uid),
                "feedback_type": request.feedback_type,
                "comment": request.comment,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            feedback_ref.set(feedback_data)
            logger.info(f"Feedback created for message {message_id[:8]}...")
            
            return {
                "message": "Thank you for your feedback!",
                "feedback_type": request.feedback_type
            }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )

@router.get("/feedback/bulk")
async def get_bulk_feedback(
    message_ids: str,  # Comma-separated list of message IDs
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Get feedback status for multiple messages.
    """
    try:
        db = get_firestore_client()
        
        # Parse message IDs
        ids = [id.strip() for id in message_ids.split(",") if id.strip()]
        
        feedback_status = {}
        
        for message_id in ids:
            feedback_doc_id = f"{user_uid}_{message_id}"
            feedback_ref = db.collection("feedback").document(feedback_doc_id)
            feedback_doc = feedback_ref.get()
            
            if feedback_doc.exists:
                feedback_status[message_id] = feedback_doc.to_dict().get("feedback_type")
            else:
                feedback_status[message_id] = None
        
        return {"feedback": feedback_status}
        
    except Exception as e:
        logger.error(f"Error fetching feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch feedback"
        )