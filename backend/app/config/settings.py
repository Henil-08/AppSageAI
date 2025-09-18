"""Configuration settings for AppSageAI backend."""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from pathlib import Path
class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Application
    app_name: str = Field(default="AppSageAI", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    
    # CORS
    cors_origin_1: str = Field(default="http://localhost:3000", description="CORS origin 1")
    cors_origin_2: str = Field(default="", description="CORS origin 2")
    cors_origin_3: str = Field(default="", description="CORS origin 3")

    
    @property
    def cors_origins(self) -> List[str]:
        """Build CORS origins list from individual env vars."""
        origins = []
        if self.cors_origin_1:
            origins.append(self.cors_origin_1)
        if self.cors_origin_2:
            origins.append(self.cors_origin_2)
        if self.cors_origin_3:
            origins.append(self.cors_origin_3)
        
        # Default fallback for development
        if not origins:
            origins = ["http://localhost:3000"]
        
        return origins
    
    # Firebase/Google Cloud
    gcp_project_id: str = Field(default="appsageai-472321", description="GCP Project ID")
    firebase_service_account_path: Path = Field(
        default=Path("./service-account.json"),
        description="Path to Firebase service account JSON"
    )
    firestore_database: str = Field(
        default="appsageai",
        description="Firestore database name"
    )
    
    # LLM Configuration
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    model_name: str = Field(default="llama-3.3-70b-versatile", description="LLM model name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    
    # HuggingFace
    hf_token: Optional[str] = Field(default=None, description="HuggingFace token")
    
    # Langfuse Configuration
    langfuse_enabled: bool = Field(default=False, description="Enable Langfuse monitoring")
    langfuse_public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    langfuse_secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host"
    )
    trace_pii: bool = Field(default=False, description="Trace PII data in Langfuse")
    
    # Security
    jwt_secret_key: str = Field(
        default="change-this-in-production",
        description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=60,
        description="Access token expiration in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=30,
        description="Refresh token expiration in days"
    )
    
    # Encryption
    encryption_key: Optional[str] = Field(
        default=None,
        description="Encryption key for additional security"
    )
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Max requests per period")
    rate_limit_period: int = Field(default=60, description="Rate limit period in seconds")
    
    # Document Processing
    max_file_size_mb: int = Field(default=10, description="Max file size in MB")
    allowed_file_type: str = Field(default="pdf", description="Allowed file type")
    
    @property
    def allowed_file_types(self) -> List[str]:
        """Return allowed file types as a list for compatibility."""
        return [self.allowed_file_type]
    
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    
    @field_validator("firebase_service_account_path", mode="before")
    @classmethod
    def validate_service_account_path(cls, v):
        if isinstance(v, str):
            v = Path(v)
        if not v.exists():
            # In production, this should fail. In dev, we can create a dummy
            print(f"Warning: Service account file not found at {v}")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"
    
    def get_groq_api_key(self) -> str:
        """Get Groq API key from environment or Secret Manager."""
        if self.groq_api_key:
            return self.groq_api_key
        
        if self.is_production:
            # In production, fetch from Secret Manager
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.gcp_project_id}/secrets/groq-api-key/versions/latest"
            try:
                response = client.access_secret_version(request={"name": name})
                return response.payload.data.decode("UTF-8")
            except Exception as e:
                print(f"Error fetching secret: {e}")
                raise ValueError("Groq API key not found in Secret Manager")
        
        raise ValueError("Groq API key not configured")


# Create settings instance
settings = Settings()