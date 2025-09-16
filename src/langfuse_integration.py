"""Langfuse integration for privacy-preserving monitoring."""

import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from logger import logger

class PrivacyLangfuseHandler:
    """Custom Langfuse handler that respects privacy."""
    
    def __init__(self, 
                 public_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 host: Optional[str] = None,
                 trace_pii: bool = False):
        """
        Initialize Langfuse with privacy settings.
        
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
            trace_pii: Whether to trace PII data (default: False)
        """
        self.trace_pii = trace_pii
        self.langfuse = None
        self.callback_handler = None
        
        if public_key and secret_key:
            try:
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                logger.info("Langfuse initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
    
    def create_handler(self, session_id: str, user_id: Optional[str] = None) -> Optional[CallbackHandler]:
        """
        Create a callback handler for a session.
        
        Args:
            session_id: Current session ID
            user_id: User identifier (will be hashed)
        
        Returns:
            CallbackHandler or None if Langfuse is not initialized
        """
        if not self.langfuse:
            return None
        
        # Hash user ID for privacy
        hashed_user_id = None
        if user_id:
            hashed_user_id = hashlib.sha256(user_id.lower().encode()).hexdigest()[:16]
        
        try:
            self.callback_handler = CallbackHandler(
                user_id=hashed_user_id,
                session_id=session_id,
                trace_name=f"appsageai-session-{session_id[:8]}"
            )
            return self.callback_handler
        except Exception as e:
            logger.error(f"Failed to create Langfuse handler: {e}")
            return None
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text to remove PII if necessary.
        
        Args:
            text: Input text to sanitize
        
        Returns:
            Sanitized text
        """
        if self.trace_pii:
            return text
        
        # Simple PII removal (in production, use more sophisticated methods)
        # This is a placeholder - implement proper PII detection/removal
        sanitized = text
        
        # Remove email addresses
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        sanitized = re.sub(email_pattern, '[EMAIL]', sanitized)
        
        # Remove phone numbers (simple pattern)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        sanitized = re.sub(phone_pattern, '[PHONE]', sanitized)
        
        # Remove SSN-like patterns
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        sanitized = re.sub(ssn_pattern, '[SSN]', sanitized)
        
        return sanitized
    
    def log_generation(self,
                       name: str,
                       input_text: str,
                       output_text: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       duration_ms: Optional[int] = None,
                       tokens_used: Optional[int] = None,
                       model: Optional[str] = None):
        """
        Log a generation event to Langfuse.
        
        Args:
            name: Name of the generation
            input_text: Input text (will be sanitized)
            output_text: Output text (will be sanitized)
            metadata: Additional metadata
            duration_ms: Response time in milliseconds
            tokens_used: Number of tokens used
            model: Model name
        """
        if not self.langfuse:
            return
        
        try:
            # Prepare metadata
            meta = metadata or {}
            meta.update({
                'duration_ms': duration_ms,
                'tokens_used': tokens_used,
                'model': model,
                'timestamp': datetime.now().isoformat(),
                'privacy_mode': not self.trace_pii
            })
            
            # Sanitize inputs if needed
            safe_input = self.sanitize_input(input_text) if not self.trace_pii else input_text
            safe_output = self.sanitize_input(output_text) if not self.trace_pii else output_text
            
            # Create generation
            generation = self.langfuse.generation(
                name=name,
                input=safe_input,
                output=safe_output,
                metadata=meta,
                model=model,
                usage={
                    "total_tokens": tokens_used,
                } if tokens_used else None,
                latency=duration_ms / 1000 if duration_ms else None
            )
            
            logger.debug(f"Logged generation to Langfuse: {name}")
            
        except Exception as e:
            logger.error(f"Failed to log to Langfuse: {e}")
    
    def log_feedback(self,
                    trace_id: str,
                    observation_id: str,
                    feedback_type: str,
                    score: Optional[float] = None,
                    comment: Optional[str] = None):
        """
        Log user feedback to Langfuse.
        
        Args:
            trace_id: Trace ID
            observation_id: Observation ID
            feedback_type: Type of feedback (thumbs_up/thumbs_down)
            score: Numeric score (optional)
            comment: User comment (optional)
        """
        if not self.langfuse:
            return
        
        try:
            score_value = 1.0 if feedback_type == 'thumbs_up' else 0.0
            
            self.langfuse.score(
                trace_id=trace_id,
                observation_id=observation_id,
                name="user_feedback",
                value=score_value,
                comment=self.sanitize_input(comment) if comment and not self.trace_pii else comment
            )
            
            logger.debug(f"Logged feedback to Langfuse: {feedback_type}")
            
        except Exception as e:
            logger.error(f"Failed to log feedback to Langfuse: {e}")
    
    def flush(self):
        """Flush any pending events to Langfuse."""
        if self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")
    
    def create_trace_config(self, 
                           session_id: str, 
                           user_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a configuration dict for LangChain with Langfuse callback.
        
        Args:
            session_id: Session ID
            user_id: User ID (will be hashed)
            metadata: Additional metadata
        
        Returns:
            Configuration dictionary for LangChain
        """
        handler = self.create_handler(session_id, user_id)
        
        if not handler:
            return {}
        
        run_id = str(uuid.uuid4())
        
        config = {
            "callbacks": [handler],
            "run_id": run_id,
            "metadata": metadata or {},
            "tags": ["appsageai", "v2"]
        }
        
        return config