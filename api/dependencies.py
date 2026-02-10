"""
Dependency Injection Functions
==============================

FastAPI dependency injection for pipeline, job manager, and authentication.
Implemented in Phase 2.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError

from config import get_settings
from core.pipeline import MedicalDocumentationPipeline

# Security scheme for JWT Bearer tokens
security = HTTPBearer(auto_error=False)


def get_pipeline() -> MedicalDocumentationPipeline:
    """
    Dependency to get the pipeline instance from app state.
    
    The pipeline is pre-loaded during application startup (lifespan)
    to avoid model loading delays on first request.
    
    Returns:
        MedicalDocumentationPipeline: The configured pipeline instance
        
    Raises:
        HTTPException: If pipeline is not initialized
    """
    from api.main import app_state
    
    pipeline = app_state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized. Service is starting up."
        )
    return pipeline


def get_job_manager():
    """
    Dependency to get job manager instance.
    
    Creates a singleton JobManager if not already in app state.
    JobManager handles Redis-based job queue operations.
    
    Returns:
        JobManager: The job manager instance
    """
    from api.main import app_state
    from api.services.job_manager import JobManager
    
    if "job_manager" not in app_state:
        app_state["job_manager"] = JobManager()
    return app_state["job_manager"]


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency to get current authenticated user from JWT token.
    
    Validates the Bearer token and extracts user information.
    Use this dependency for routes that REQUIRE authentication.
    
    Args:
        credentials: HTTP Authorization header with Bearer token
        
    Returns:
        dict: User information with 'user_id' and 'email'
        
    Raises:
        HTTPException: 401 if token is missing or invalid
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    
    try:
        from api.utils.security import verify_token
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return {
            "user_id": user_id,
            "email": payload.get("email")
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """
    Optional authentication - returns None if not authenticated.
    
    Use this dependency for routes where authentication is optional
    but provides additional features when present.
    
    Args:
        credentials: Optional HTTP Authorization header
        
    Returns:
        dict: User information if authenticated, None otherwise
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
