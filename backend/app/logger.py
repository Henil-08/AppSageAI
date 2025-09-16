"""Logging configuration for AppSageAI backend."""

import os
import sys
import logging
from pathlib import Path

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Logging configuration
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_filepath = log_dir / "running_logs.log"

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger
logger = logging.getLogger(__name__)

# Add custom privacy filter to redact sensitive information
class PrivacyFilter(logging.Filter):
    """Filter to redact sensitive information from logs."""
    
    def filter(self, record):
        # List of sensitive keys to redact
        sensitive_keys = ['password', 'token', 'secret', 'api_key', 'resume_content', 'chat_message']
        
        # Redact sensitive information in log messages
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for key in sensitive_keys:
                if key in msg.lower():
                    # Simple redaction - enhance as needed
                    msg = msg.replace(key, f"{key}=REDACTED")
            record.msg = msg
        
        return True

# Add privacy filter to all handlers
for handler in logging.getLogger().handlers:
    handler.addFilter(PrivacyFilter())

# Log startup
logger.info("AppSageAI Backend Logger initialized")
logger.info(f"Log file: {log_filepath}")

# Suppress noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)