"""Custom prompts API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel

from app.auth.firebase import get_current_user_uid, get_firestore_client
from app.services.encryption import encryption_service
from app.db.models import AnalysisType
from app.core.prompts import PromptManager
from app.logger import logger

router = APIRouter()

# Request models
class UpdatePromptRequest(BaseModel):
    analysis_type: str
    prompt_template: str

class UpdateAllPromptsRequest(BaseModel):
    prompts: Dict[str, str]

# Initialize default prompt manager
default_prompt_manager = PromptManager()

@router.get("/defaults")
async def get_default_prompts(
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Get all default prompt templates.
    """
    try:
        # Get default prompts
        default_prompts = default_prompt_manager.get_all_prompts()
        
        # Convert enum keys to strings
        prompts_dict = {
            analysis_type.value: prompt 
            for analysis_type, prompt in default_prompts.items()
        }
        
        return {
            "prompts": prompts_dict,
            "message": "Default prompts retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting default prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get default prompts"
        )

@router.get("/custom")
async def get_custom_prompts(
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Get user's custom prompt templates (decrypted).
    """
    try:
        db = get_firestore_client()
        
        # Get user's custom prompts
        user_ref = db.collection("users").document(user_uid)
        prompts_ref = user_ref.collection("custom_prompts")
        
        custom_prompts = {}
        has_custom = False
        
        for doc in prompts_ref.stream():
            prompt_data = doc.to_dict()
            encrypted_prompt = prompt_data.get("prompt_template", "")
            
            # Decrypt the prompt
            try:
                # Try to decrypt assuming it's encrypted
                decrypted_prompt = encryption_service.decrypt_content(encrypted_prompt).decode()
                custom_prompts[doc.id] = decrypted_prompt
                has_custom = True
            except Exception as decrypt_error:
                # If decryption fails, it might be old unencrypted data
                # Check if it looks like plain text (contains common words)
                if isinstance(encrypted_prompt, str) and any(word in encrypted_prompt.lower() for word in ['you', 'the', 'resume', 'candidate']):
                    # It's likely unencrypted old data
                    custom_prompts[doc.id] = encrypted_prompt
                    has_custom = True
                    
                    # Re-encrypt it for future security
                    try:
                        prompt_ref = prompts_ref.document(doc.id)
                        prompt_ref.update({
                            "prompt_template": encryption_service.encrypt_content(encrypted_prompt.encode()),
                            "updated_at": datetime.now(timezone.utc)
                        })
                        logger.info(f"Re-encrypted old prompt for user {user_uid[:8]}... - Type: {doc.id}")
                    except:
                        pass
                else:
                    logger.warning(f"Could not decrypt prompt for {doc.id}: {decrypt_error}")
        
        # If user has no custom prompts, return defaults
        if not custom_prompts:
            default_prompts = default_prompt_manager.get_all_prompts()
            custom_prompts = {
                analysis_type.value: prompt 
                for analysis_type, prompt in default_prompts.items()
            }
        
        return {
            "prompts": custom_prompts,
            "has_custom": has_custom,
            "message": "Custom prompts retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting custom prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get custom prompts"
        )

@router.post("/update")
async def update_prompt(
    request: UpdatePromptRequest,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Update a single custom prompt template (encrypted).
    """
    try:
        db = get_firestore_client()
        
        # Validate analysis type
        try:
            analysis_type = AnalysisType(request.analysis_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid analysis type: {request.analysis_type}"
            )
        
        # Validate that {context} is present (required for RAG)
        if "{context}" not in request.prompt_template:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt must include {context} placeholder for resume analysis to work"
            )
        
        # Store custom prompt (ENCRYPTED)
        user_ref = db.collection("users").document(user_uid)
        prompt_ref = user_ref.collection("custom_prompts").document(request.analysis_type)
        
        # Encrypt the prompt template
        encrypted_prompt = encryption_service.encrypt_content(request.prompt_template.encode())
        
        prompt_data = {
            "prompt_template": encrypted_prompt,
            "updated_at": datetime.now(timezone.utc),
            "analysis_type": request.analysis_type
        }
        
        prompt_ref.set(prompt_data)
        
        logger.info(f"Custom prompt updated (encrypted) for user {user_uid[:8]}... - Type: {request.analysis_type}")
        
        return {
            "message": "Prompt updated successfully",
            "analysis_type": request.analysis_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update prompt"
        )

@router.post("/update-all")
async def update_all_prompts(
    request: UpdateAllPromptsRequest,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Update all custom prompt templates at once (encrypted).
    """
    try:
        db = get_firestore_client()
        
        user_ref = db.collection("users").document(user_uid)
        updated_count = 0
        
        # Update each prompt
        for analysis_type, prompt_template in request.prompts.items():
            # Validate analysis type
            try:
                AnalysisType(analysis_type)
            except ValueError:
                logger.warning(f"Skipping invalid analysis type: {analysis_type}")
                continue
            
            # Ensure {context} is present
            if "{context}" not in prompt_template:
                # Auto-add {context} if missing
                prompt_template += "\n\nContext from resume:\n{context}"
                logger.warning(f"Auto-added {{context}} to prompt for {analysis_type}")
            
            prompt_ref = user_ref.collection("custom_prompts").document(analysis_type)
            
            # Encrypt the prompt template
            encrypted_prompt = encryption_service.encrypt_content(prompt_template.encode())
            
            prompt_data = {
                "prompt_template": encrypted_prompt,
                "updated_at": datetime.now(timezone.utc),
                "analysis_type": analysis_type
            }
            
            prompt_ref.set(prompt_data)
            updated_count += 1
        
        logger.info(f"All custom prompts updated (encrypted) for user {user_uid[:8]}... - Count: {updated_count}")
        
        return {
            "message": "All prompts updated successfully",
            "updated_count": updated_count
        }
        
    except Exception as e:
        logger.error(f"Error updating prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update prompts"
        )

@router.post("/reset/{analysis_type}")
async def reset_prompt_to_default(
    analysis_type: str,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Reset a specific prompt to default.
    """
    try:
        db = get_firestore_client()
        
        # Validate analysis type
        try:
            analysis_type_enum = AnalysisType(analysis_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid analysis type: {analysis_type}"
            )
        
        # Delete custom prompt
        user_ref = db.collection("users").document(user_uid)
        prompt_ref = user_ref.collection("custom_prompts").document(analysis_type)
        
        # Check if it exists before deleting
        if prompt_ref.get().exists:
            prompt_ref.delete()
            logger.info(f"Prompt reset to default for user {user_uid[:8]}... - Type: {analysis_type}")
        
        # Return the default prompt
        default_prompt = default_prompt_manager.get_prompt(analysis_type_enum)
        
        return {
            "message": "Prompt reset to default",
            "analysis_type": analysis_type,
            "default_prompt": default_prompt
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset prompt"
        )

@router.post("/reset-all")
async def reset_all_prompts_to_default(
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Reset all prompts to default by deleting all custom prompts.
    """
    try:
        db = get_firestore_client()
        
        # Delete all custom prompts
        user_ref = db.collection("users").document(user_uid)
        prompts_ref = user_ref.collection("custom_prompts")
        
        deleted_count = 0
        # Delete all documents in the collection
        for doc in prompts_ref.stream():
            doc.reference.delete()
            deleted_count += 1
        
        logger.info(f"All prompts reset to default for user {user_uid[:8]}... - Deleted: {deleted_count}")
        
        return {
            "message": "All prompts reset to default successfully",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error resetting prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset prompts"
        )

@router.get("/validate/{analysis_type}")
async def validate_prompt(
    analysis_type: str,
    user_uid: str = Depends(get_current_user_uid)
) -> Dict[str, Any]:
    """
    Validate if a custom prompt exists and is properly formatted.
    """
    try:
        db = get_firestore_client()
        
        # Validate analysis type
        try:
            AnalysisType(analysis_type)
        except ValueError:
            return {
                "valid": False,
                "message": f"Invalid analysis type: {analysis_type}"
            }
        
        # Check if custom prompt exists
        user_ref = db.collection("users").document(user_uid)
        prompt_ref = user_ref.collection("custom_prompts").document(analysis_type)
        
        prompt_doc = prompt_ref.get()
        if not prompt_doc.exists:
            return {
                "valid": True,
                "uses_default": True,
                "message": "Using default prompt"
            }
        
        # Try to decrypt and validate
        try:
            encrypted_prompt = prompt_doc.to_dict().get("prompt_template")
            decrypted_prompt = encryption_service.decrypt_content(encrypted_prompt).decode()
            
            # Check for required {context} placeholder
            has_context = "{context}" in decrypted_prompt
            
            return {
                "valid": has_context,
                "uses_default": False,
                "has_context": has_context,
                "message": "Custom prompt is valid" if has_context else "Custom prompt missing {context} placeholder"
            }
            
        except Exception as e:
            logger.error(f"Error validating prompt: {e}")
            return {
                "valid": False,
                "uses_default": False,
                "message": "Error decrypting custom prompt"
            }
        
    except Exception as e:
        logger.error(f"Error validating prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate prompt"
        )