"""
Rate Limiting Middleware
========================

Rate limiting setup using slowapi.
Implemented in Phase 2 (basic setup), enhanced in Phase 8.
"""

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import get_settings


# Create limiter instance with IP-based rate limiting
limiter = Limiter(key_func=get_remote_address)


def setup_rate_limiting(app: FastAPI) -> None:
    """
    Set up rate limiting for the FastAPI application.
    
    Attaches the limiter to the app state and registers
    the exception handler for rate limit exceeded errors.
    
    Args:
        app: The FastAPI application instance
        
    Usage in routes:
        from api.middleware.rate_limiter import limiter
        
        @router.post("/process")
        @limiter.limit("10/minute")
        async def submit_job(request: Request, ...):
            ...
    """
    settings = get_settings()
    
    # Attach limiter to app state for access in routes
    app.state.limiter = limiter
    
    # Register exception handler for rate limit exceeded
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
