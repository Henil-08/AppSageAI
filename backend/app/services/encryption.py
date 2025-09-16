"""Encryption service for additional server-side security."""

import hashlib
import secrets
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from app.logger import logger
from app.config.settings import settings


class EncryptionService:
    """
    Service for server-side encryption.
    Note: Primary encryption happens client-side. This is an additional layer.
    """
    
    def __init__(self):
        """Initialize encryption service."""
        self.cipher = self._initialize_cipher()
    
    def _initialize_cipher(self) -> Optional[Fernet]:
        """Initialize Fernet cipher."""
        if not settings.encryption_key:
            # Generate a new key if not provided (development only)
            if settings.is_development:
                key = Fernet.generate_key()
                logger.warning("Generated temporary encryption key for development")
                return Fernet(key)
            else:
                logger.warning("No encryption key configured")
                return None
        
        # Use provided key
        try:
            # Derive key from password if it's not a valid Fernet key
            if len(settings.encryption_key) != 44:  # Fernet keys are 44 chars
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'appsageai-salt',  # In production, use unique salt
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(
                    kdf.derive(settings.encryption_key.encode())
                )
                return Fernet(key)
            else:
                return Fernet(settings.encryption_key.encode())
        except Exception as e:
            logger.error(f"Failed to initialize cipher: {e}")
            return None
    
    def encrypt_metadata(self, data: str) -> str:
        """
        Encrypt metadata (not user content - that's encrypted client-side).
        
        Args:
            data: String to encrypt
        
        Returns:
            Encrypted string
        """
        if not self.cipher:
            return data  # Return as-is if encryption not available
        
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_metadata(self, encrypted_data: str) -> str:
        """
        Decrypt metadata.
        
        Args:
            encrypted_data: Encrypted string
        
        Returns:
            Decrypted string
        """
        if not self.cipher:
            return encrypted_data
        
        try:
            decrypted = self.cipher.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    @staticmethod
    def hash_identifier(identifier: str) -> str:
        """
        Create a one-way hash of an identifier for privacy.
        
        Args:
            identifier: String to hash (email, user ID, etc.)
        
        Returns:
            Hashed identifier
        """
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    @staticmethod
    def hash_file(content: bytes) -> str:
        """
        Create hash of file content for deduplication.
        
        Args:
            content: File content
        
        Returns:
            File hash
        """
        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    def generate_id() -> str:
        """Generate a secure random ID."""
        return secrets.token_urlsafe(16)
    
    @staticmethod
    def sanitize_for_logging(text: str) -> str:
        """
        Sanitize text for logging by removing potential PII.
        
        Args:
            text: Text to sanitize
        
        Returns:
            Sanitized text
        """
        import re
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '[EMAIL]', text)
        
        # Phone pattern (US)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        text = re.sub(phone_pattern, '[PHONE]', text)
        
        # SSN pattern
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        text = re.sub(ssn_pattern, '[SSN]', text)
        
        # Credit card pattern (basic)
        cc_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        text = re.sub(cc_pattern, '[CREDIT_CARD]', text)
        
        return text


# Global encryption service instance
encryption_service = EncryptionService()