"""
Global Error Handler Middleware
================================

Maps custom exceptions to HTTP status codes and formats error responses.
Implemented in Phase 2.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse

from exceptions import (
    MedScribeError,
    AudioFileNotFoundError,
    UnsupportedAudioFormatError,
    AudioTooLongError,
    OllamaConnectionError,
    ConfigurationError,
    TranscriptionError,
    GenerationError,
)
from config import get_settings


# Map exceptions to HTTP status codes
EXCEPTION_STATUS_MAP = {
    AudioFileNotFoundError: status.HTTP_404_NOT_FOUND,
    UnsupportedAudioFormatError: status.HTTP_400_BAD_REQUEST,
    AudioTooLongError: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    OllamaConnectionError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    TranscriptionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    GenerationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


async def error_handler_middleware(request: Request, call_next):
    """
    Global error handling middleware.
    
    Catches MedScribe exceptions and converts them to appropriate
    HTTP responses with structured error bodies.
    
    Args:
        request: The incoming request
        call_next: The next middleware/route handler
        
    Returns:
        Response or JSONResponse with error details
    """
    try:
        response = await call_next(request)
        return response
    except MedScribeError as e:
        # Use existing to_dict() method from our exception hierarchy
        status_code = EXCEPTION_STATUS_MAP.get(
            type(e), 
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        return JSONResponse(
            status_code=status_code,
            content=e.to_dict()
        )
    except Exception as e:
        # Unexpected errors - hide details in production
        settings = get_settings()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_type": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": {"error": str(e)} if settings.api_debug else {}
            }
        )
