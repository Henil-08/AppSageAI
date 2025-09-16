"""Firebase authentication and initialization."""

from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, auth, firestore
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import google.cloud.firestore

from app.logger import logger
from app.config.settings import settings


# Initialize Firebase Admin SDK
_firebase_app = None
_firestore_client = None

security = HTTPBearer()


def initialize_firebase():
    """Initialize Firebase Admin SDK."""
    global _firebase_app, _firestore_client
    
    if _firebase_app is not None:
        logger.info("Firebase already initialized")
        return _firebase_app
    
    try:
        # Load service account
        cred = credentials.Certificate(str(settings.firebase_service_account_path))
        
        # Initialize app with correct database
        _firebase_app = firebase_admin.initialize_app(
            cred,
            {
                'projectId': settings.gcp_project_id,
                'databaseURL': f'https://{settings.gcp_project_id}.firebaseio.com'
            }
        )
        
        # Initialize Firestore client with correct database
        # Use the named database instead of default
        _firestore_client = firestore.client(app=_firebase_app, database_id=settings.firestore_database)
        
        logger.info(f"Firebase initialized for project: {settings.gcp_project_id}, database: {settings.firestore_database}")
        return _firebase_app
        
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        raise


def get_firestore_client() -> google.cloud.firestore.Client:
    """Get Firestore client instance."""
    global _firestore_client
    
    if _firestore_client is None:
        initialize_firebase()
    
    return _firestore_client


async def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Verify Firebase ID token from Authorization header.
    
    Args:
        credentials: Bearer token from Authorization header
    
    Returns:
        Decoded token with user information
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    
    try:
        # Verify the token
        decoded_token = auth.verify_id_token(token)
        
        # Log successful authentication (without PII)
        logger.info(f"User authenticated: {decoded_token['uid'][:8]}...")
        
        return decoded_token
        
    except auth.InvalidIdTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.ExpiredIdTokenError:
        logger.warning("Expired token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    token: Dict[str, Any] = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get current authenticated user information.
    
    Args:
        token: Decoded Firebase token
    
    Returns:
        User information dictionary
    """
    user_info = {
        "uid": token.get("uid"),
        "email": token.get("email"),
        "email_verified": token.get("email_verified", False),
        "name": token.get("name"),
        "picture": token.get("picture"),
        "provider": token.get("firebase", {}).get("sign_in_provider"),
        "auth_time": token.get("auth_time"),
    }
    
    # Remove None values
    user_info = {k: v for k, v in user_info.items() if v is not None}
    
    return user_info


async def get_current_user_uid(
    token: Dict[str, Any] = Depends(verify_firebase_token)
) -> str:
    """
    Get current user's UID.
    
    Args:
        token: Decoded Firebase token
    
    Returns:
        User UID
    """
    return token["uid"]


class UserPermissions:
    """User permission checker."""
    
    def __init__(self, required_email_verified: bool = False):
        self.required_email_verified = required_email_verified
    
    async def __call__(
        self,
        user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """
        Check user permissions.
        
        Args:
            user: Current user information
        
        Returns:
            User information if permissions are satisfied
        
        Raises:
            HTTPException: If permissions are not satisfied
        """
        if self.required_email_verified and not user.get("email_verified", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Email verification required"
            )
        
        return user


# Optional: Admin check
async def require_admin(
    user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require admin privileges.
    
    Args:
        user: Current user information
    
    Returns:
        User information if admin
    
    Raises:
        HTTPException: If user is not admin
    """
    # Check if user is admin (you can implement your own logic)
    # For now, we'll check if email matches admin emails
    admin_emails = ["admin@appsageai.com"]  # Add your admin emails
    
    if user.get("email") not in admin_emails:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return user


# Helper function to get or create user document
async def get_or_create_user_doc(
    uid: str,
    email: Optional[str] = None,
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get or create user document in Firestore.
    
    Args:
        uid: User UID
        email: User email
        name: User name
    
    Returns:
        User document data
    """
    db = get_firestore_client()
    user_ref = db.collection("users").document(uid)
    
    # Try to get existing user
    user_doc = user_ref.get()
    
    if user_doc.exists:
        return user_doc.to_dict()
    
    # Create new user document
    user_data = {
        "uid": uid,
        "email": email,
        "name": name,
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "plan": "free",
        "usage": {
            "analyses_count": 0,
            "tokens_used": 0,
            "last_analysis": None
        }
    }
    
    # Remove None values
    user_data = {k: v for k, v in user_data.items() if v is not None}
    
    user_ref.set(user_data)
    logger.info(f"Created new user document for {uid[:8]}...")
    
    return user_data