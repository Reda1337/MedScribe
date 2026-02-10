"""
FastAPI Main Application
========================

Main FastAPI application instance with middleware, routes, and lifespan management.
Implemented in Phase 2.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from api.middleware.error_handler import error_handler_middleware
from api.middleware.rate_limiter import setup_rate_limiting
from api.routes import health, jobs, auth, websocket
from config import get_settings
from core.pipeline import create_pipeline


# Global application state - stores pipeline and other singletons
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    
    Startup:
    - Pre-loads the ML pipeline and models to avoid first-request delays
    - Stores references in app_state for dependency injection
    
    Shutdown:
    - Cleans up resources and clears state
    
    This pattern ensures models are loaded once at startup rather than
    on each request, significantly improving response times.
    """
    settings = get_settings()
    
    # Startup: Pre-load pipeline and models
    print("🚀 Starting MedScribe AI API...")
    print(f"   Loading Whisper model: {settings.whisper_model}")
    print(f"   Ollama model: {settings.ollama_model}")
    
    try:
        # Create and store pipeline instance
        pipeline = create_pipeline(settings)
        app_state["pipeline"] = pipeline
        app_state["settings"] = settings
        
        print("✅ Pipeline loaded successfully")
        print(f"📍 API running at http://{settings.api_host}:{settings.api_port}")
        print(f"📚 Docs available at http://{settings.api_host}:{settings.api_port}/api/docs")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        # Store None - health check will report unhealthy
        app_state["pipeline"] = None
        app_state["settings"] = settings
    
    yield  # Application runs here
    
    # Shutdown: Cleanup
    print("\n🛑 Shutting down MedScribe AI API...")
    app_state.clear()
    print("✅ Cleanup complete")


# Create FastAPI application
app = FastAPI(
    title="MedScribe AI API",
    description="""
    Medical documentation system - Convert audio recordings to SOAP notes.
    
    ## Features
    - Audio transcription with Whisper
    - Speaker diarization (Doctor/Patient identification)
    - SOAP note generation with LLM
    - Real-time progress via WebSocket
    - Background job processing
    
    ## Authentication
    Most endpoints require JWT Bearer token authentication.
    Register and login to obtain tokens.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# =============================================================================
# Middleware Setup (order matters - first added = outermost)
# =============================================================================

# Get settings for middleware configuration
settings = get_settings()

# CORS middleware - allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Host middleware - security against host header attacks
# In production, set this to your actual domains
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production: ["api.example.com"]
)

# Global error handling middleware
app.middleware("http")(error_handler_middleware)

# Rate limiting setup
setup_rate_limiting(app)


# =============================================================================
# Router Registration
# =============================================================================

# Health check endpoints (no auth required)
app.include_router(
    health.router,
    prefix="/api/v1",
    tags=["health"]
)

# Job management endpoints
app.include_router(
    jobs.router,
    prefix="/api/v1",
    tags=["jobs"]
)

# Authentication endpoints
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["auth"]
)

# WebSocket endpoints for real-time updates
app.include_router(
    websocket.router,
    prefix="/api/v1",
    tags=["websocket"]
)


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint - API information and links.
    
    Returns basic information about the API and links to documentation.
    """
    return {
        "message": "MedScribe AI API",
        "description": "Medical documentation system - Convert audio to SOAP notes",
        "version": "1.0.0",
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "health": "/api/v1/health"
    }
