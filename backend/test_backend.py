"""Test script to verify backend setup."""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.logger import logger
from app.config.settings import settings
from app.services.encryption import encryption_service


def test_configuration():
    """Test configuration loading."""
    print("\n🔧 Testing Configuration...")
    print(f"  App Name: {settings.app_name}")
    print(f"  Environment: {settings.environment}")
    print(f"  GCP Project: {settings.gcp_project_id}")
    print(f"  API Port: {settings.api_port}")
    print(f"  CORS Origins: {settings.cors_origins}")
    
    # Check if service account exists
    if settings.firebase_service_account_path.exists():
        print(f"  ✅ Service account found: {settings.firebase_service_account_path}")
    else:
        print(f"  ❌ Service account NOT found: {settings.firebase_service_account_path}")
    
    # Check if Groq API key is set
    if settings.groq_api_key:
        print(f"  ✅ Groq API key configured")
    else:
        print(f"  ⚠️  Groq API key not set")


def test_logger():
    """Test logging."""
    print("\n📝 Testing Logger...")
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Check if log file was created
    log_file = Path("logs/running_logs.log")
    if log_file.exists():
        print(f"  ✅ Log file created: {log_file}")
        # Read last line
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"  Last log: {lines[-1].strip()}")
    else:
        print(f"  ❌ Log file not created")


def test_encryption():
    """Test encryption service."""
    print("\n🔐 Testing Encryption...")
    
    # Test hashing
    test_email = "user@example.com"
    hashed = encryption_service.hash_identifier(test_email)
    print(f"  Hash of '{test_email}': {hashed}")
    
    # Test ID generation
    new_id = encryption_service.generate_id()
    print(f"  Generated ID: {new_id}")
    
    # Test metadata encryption
    if encryption_service.cipher:
        test_data = "sensitive metadata"
        encrypted = encryption_service.encrypt_metadata(test_data)
        decrypted = encryption_service.decrypt_metadata(encrypted)
        print(f"  Original: '{test_data}'")
        print(f"  Encrypted: '{encrypted[:30]}...'")
        print(f"  Decrypted: '{decrypted}'")
        print(f"  ✅ Encryption working: {test_data == decrypted}")
    else:
        print(f"  ⚠️  Encryption not configured")
    
    # Test PII sanitization
    test_text = "Contact me at user@example.com or 555-123-4567"
    sanitized = encryption_service.sanitize_for_logging(test_text)
    print(f"  PII Sanitization:")
    print(f"    Original: {test_text}")
    print(f"    Sanitized: {sanitized}")


async def test_firebase():
    """Test Firebase initialization."""
    print("\n🔥 Testing Firebase...")
    
    try:
        from app.auth.firebase import initialize_firebase, get_firestore_client
        
        # Initialize Firebase
        initialize_firebase()
        print("  ✅ Firebase initialized successfully")
        
        # Test Firestore connection
        db = get_firestore_client()
        
        # Try to read a collection (won't fail even if empty)
        test_collection = db.collection("_test").limit(1).get()
        print("  ✅ Firestore connection successful")
        
    except Exception as e:
        print(f"  ❌ Firebase error: {e}")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("🚀 AppSageAI Backend Test Suite")
    print("=" * 50)
    
    # Run tests
    test_configuration()
    test_logger()
    test_encryption()
    await test_firebase()
    
    print("\n" + "=" * 50)
    print("✨ Tests completed!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. If all tests passed, run the server:")
    print("   uvicorn app.main:app --reload --port 8000")
    print("\n2. Then visit:")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")


if __name__ == "__main__":
    asyncio.run(main())