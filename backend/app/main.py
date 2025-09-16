"""Main FastAPI application for AppSageAI backend."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

from app.logger import logger
from app.config.settings import settings
from app.auth.firebase import initialize_firebase
from app.api import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize Firebase
    try:
        initialize_firebase()
        logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        if settings.is_production:
            raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Gzip middleware for compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Add request ID and logging middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())[:8]
    
    # Store request ID in request state
    request.state.request_id = request_id
    
    # Log request
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    # Process request and measure time
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        logger.info(f"[{request_id}] Completed in {process_time:.3f}s - Status: {response.status_code}")
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[{request_id}] Error after {process_time:.3f}s: {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id
            }
        )


# Rate limiting middleware (simple in-memory implementation)
from collections import defaultdict
from datetime import datetime, timedelta

rate_limit_storage = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware."""
    if not settings.rate_limit_enabled:
        return await call_next(request)
    
    # Skip rate limiting for health checks and docs
    if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # Get client IP
    client_ip = request.client.host
    
    # Clean old entries
    now = datetime.now()
    cutoff = now - timedelta(seconds=settings.rate_limit_period)
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if timestamp > cutoff
    ]
    
    # Check rate limit
    if len(rate_limit_storage[client_ip]) >= settings.rate_limit_requests:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    
    return await call_next(request)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name} API",
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "Disabled in production"
    }


# Include API routes
app.include_router(api_router, prefix=settings.api_prefix)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled exception: {str(exc)}", exc_info=True)
    
    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "type": type(exc).__name__,
                "request_id": request_id
            }
        )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )