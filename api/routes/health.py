"""
Health Check Endpoints
======================

API health check endpoints for monitoring and Kubernetes probes.
Implemented in Phase 3.
"""

import logging
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from config import Settings, get_settings
from api.dependencies import get_pipeline, get_job_manager
from pipeline import MedicalDocumentationPipeline
from api.services.job_manager import JobManager


# Set up module logger
logger = logging.getLogger(__name__)

router = APIRouter()


class ServiceStatus(str, Enum):
    """Service health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthStatus(str, Enum):
    """Overall application health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceCheckResult(BaseModel):
    """Result of an individual service health check."""
    status: ServiceStatus = Field(description="Service health status")
    message: Optional[str] = Field(None, description="Status message or error details")
    latency_ms: Optional[float] = Field(None, description="Check latency in milliseconds")


class SystemMetrics(BaseModel):
    """System resource metrics."""
    cpu_percent: float = Field(description="CPU usage percentage")
    memory_percent: float = Field(description="Memory usage percentage")
    memory_available_mb: float = Field(description="Available memory in MB")
    disk_usage_percent: float = Field(description="Disk usage percentage")


class HealthCheckResponse(BaseModel):
    """Comprehensive health check response."""
    status: HealthStatus = Field(description="Overall health status")
    timestamp: str = Field(description="ISO 8601 timestamp of the health check")
    services: Dict[str, ServiceCheckResult] = Field(description="Individual service statuses")
    system_metrics: SystemMetrics = Field(description="System resource metrics")


class ReadinessResponse(BaseModel):
    """Kubernetes readiness probe response."""
    status: str = Field(description="Readiness status")
    message: str = Field(description="Status message")
    timestamp: str = Field(description="ISO 8601 timestamp")


class LivenessResponse(BaseModel):
    """Kubernetes liveness probe response."""
    status: str = Field(description="Liveness status")
    timestamp: str = Field(description="ISO 8601 timestamp")


def check_redis(job_manager: JobManager) -> ServiceCheckResult:
    """
    Check Redis connectivity and health.

    Args:
        job_manager: JobManager instance with Redis client

    Returns:
        ServiceCheckResult with Redis health status
    """
    try:
        import time
        start_time = time.time()

        if job_manager.redis_client is None:
            return ServiceCheckResult(
                status=ServiceStatus.DEGRADED,
                message="Redis not configured, using in-memory fallback"
            )

        # Test Redis with PING command
        job_manager.redis_client.ping()
        latency_ms = (time.time() - start_time) * 1000

        return ServiceCheckResult(
            status=ServiceStatus.HEALTHY,
            message="Connected",
            latency_ms=round(latency_ms, 2)
        )
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return ServiceCheckResult(
            status=ServiceStatus.UNHEALTHY,
            message=f"Connection failed: {str(e)}"
        )


def check_ollama(pipeline: MedicalDocumentationPipeline, settings: Settings) -> ServiceCheckResult:
    """
    Check Ollama LLM connectivity and model availability.

    This performs a lightweight HTTP check to Ollama's API instead of running
    full LLM inference, making health checks fast (<2s instead of 8-62s).

    Args:
        pipeline: Pipeline instance with SOAP generator
        settings: Application settings

    Returns:
        ServiceCheckResult with Ollama health status
    """
    try:
        import time
        import requests
        start_time = time.time()

        # Use Ollama's /api/tags endpoint for lightweight model availability check
        # This is much faster than running full inference (llm.invoke)
        response = requests.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=2  # 2 second timeout
        )

        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]  # Extract base name

            # Check if configured model exists (check base name, e.g., "qwen3" from "qwen3:14b")
            configured_model_base = settings.ollama_model.split(":")[0]
            model_exists = any(configured_model_base in name for name in model_names)

            if model_exists:
                latency_ms = (time.time() - start_time) * 1000
                return ServiceCheckResult(
                    status=ServiceStatus.HEALTHY,
                    message=f"Model '{settings.ollama_model}' available",
                    latency_ms=round(latency_ms, 2)
                )
            else:
                return ServiceCheckResult(
                    status=ServiceStatus.UNHEALTHY,
                    message=f"Model '{settings.ollama_model}' not found. Available: {', '.join(model_names)}. Run: ollama pull {settings.ollama_model}"
                )
        else:
            return ServiceCheckResult(
                status=ServiceStatus.UNHEALTHY,
                message=f"Ollama API returned status {response.status_code}"
            )

    except requests.exceptions.Timeout:
        return ServiceCheckResult(
            status=ServiceStatus.UNHEALTHY,
            message=f"Ollama connection timeout (2s) at {settings.ollama_base_url}"
        )
    except requests.exceptions.ConnectionError:
        logger.warning(f"Cannot connect to Ollama at {settings.ollama_base_url}")
        return ServiceCheckResult(
            status=ServiceStatus.UNHEALTHY,
            message=f"Cannot connect to Ollama at {settings.ollama_base_url}"
        )
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")
        return ServiceCheckResult(
            status=ServiceStatus.UNHEALTHY,
            message=f"Error: {str(e)}"
        )


def check_whisper(pipeline: MedicalDocumentationPipeline, settings: Settings) -> ServiceCheckResult:
    """
    Check Whisper model availability.

    Args:
        pipeline: Pipeline instance with transcriber
        settings: Application settings

    Returns:
        ServiceCheckResult with Whisper health status
    """
    try:
        # Check if transcriber is initialized
        transcriber = pipeline.transcriber

        if transcriber is None:
            return ServiceCheckResult(
                status=ServiceStatus.UNHEALTHY,
                message="Transcriber not initialized"
            )

        # Check if the model is loaded
        if hasattr(transcriber, 'model') and transcriber.model is not None:
            return ServiceCheckResult(
                status=ServiceStatus.HEALTHY,
                message=f"Model '{settings.whisper_model}' loaded on {settings.whisper_device}"
            )
        else:
            return ServiceCheckResult(
                status=ServiceStatus.DEGRADED,
                message="Model not yet loaded (lazy initialization)"
            )
    except Exception as e:
        logger.warning(f"Whisper health check failed: {e}")
        return ServiceCheckResult(
            status=ServiceStatus.UNHEALTHY,
            message=f"Error: {str(e)}"
        )


def get_system_metrics() -> SystemMetrics:
    """
    Gather system resource metrics.

    Returns:
        SystemMetrics with CPU, memory, and disk usage
    """
    try:
        # CPU usage (1 second interval for accurate reading)
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / (1024 * 1024)

        # Disk usage (root partition)
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent

        return SystemMetrics(
            cpu_percent=round(cpu_percent, 2),
            memory_percent=round(memory_percent, 2),
            memory_available_mb=round(memory_available_mb, 2),
            disk_usage_percent=round(disk_usage_percent, 2)
        )
    except Exception as e:
        logger.error(f"Failed to gather system metrics: {e}")
        # Return default metrics if gathering fails
        return SystemMetrics(
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_available_mb=0.0,
            disk_usage_percent=0.0
        )


def determine_overall_status(services: Dict[str, ServiceCheckResult]) -> HealthStatus:
    """
    Determine overall health status based on individual service statuses.

    Logic:
    - HEALTHY: All services are healthy or degraded (with graceful fallbacks)
    - DEGRADED: At least one non-critical service is unhealthy
    - UNHEALTHY: Critical services (Ollama, Whisper) are unhealthy

    Args:
        services: Dictionary of service check results

    Returns:
        Overall HealthStatus
    """
    critical_services = ["ollama", "whisper", "api"]

    # Check if any critical service is unhealthy
    for service_name in critical_services:
        if service_name in services and services[service_name].status == ServiceStatus.UNHEALTHY:
            return HealthStatus.UNHEALTHY

    # Check if any service is unhealthy or degraded
    has_unhealthy = any(s.status == ServiceStatus.UNHEALTHY for s in services.values())
    has_degraded = any(s.status == ServiceStatus.DEGRADED for s in services.values())

    if has_unhealthy:
        return HealthStatus.DEGRADED
    elif has_degraded:
        return HealthStatus.DEGRADED
    else:
        return HealthStatus.HEALTHY


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Comprehensive health check",
    description="""
    Comprehensive health check endpoint that checks all services and system metrics.

    This endpoint verifies:
    - API service availability
    - Ollama LLM connectivity and model availability
    - Redis connectivity (or in-memory fallback status)
    - Whisper model availability
    - System resource metrics (CPU, memory, disk)

    Returns HTTP 200 with detailed status information even if some services are degraded.
    Use the 'status' field to determine overall health.
    """
)
async def health_check(
    pipeline: MedicalDocumentationPipeline = Depends(get_pipeline),
    job_manager: JobManager = Depends(get_job_manager),
    settings: Settings = Depends(get_settings)
) -> HealthCheckResponse:
    """
    Perform comprehensive health check of all services.

    Args:
        pipeline: Injected pipeline instance
        job_manager: Injected job manager instance
        settings: Injected application settings

    Returns:
        HealthCheckResponse with detailed service statuses and system metrics
    """
    logger.debug("Performing comprehensive health check")

    # Check all services
    services = {
        "api": ServiceCheckResult(
            status=ServiceStatus.HEALTHY,
            message="API is running"
        ),
        "redis": check_redis(job_manager),
        "ollama": check_ollama(pipeline, settings),
        "whisper": check_whisper(pipeline, settings)
    }

    # Gather system metrics
    system_metrics = get_system_metrics()

    # Determine overall status
    overall_status = determine_overall_status(services)

    # Log overall status
    logger.info(f"Health check completed: {overall_status}")
    if overall_status != HealthStatus.HEALTHY:
        unhealthy_services = [
            name for name, check in services.items()
            if check.status != ServiceStatus.HEALTHY
        ]
        logger.warning(f"Unhealthy/degraded services: {unhealthy_services}")

    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        services=services,
        system_metrics=system_metrics
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Kubernetes readiness probe",
    description="""
    Kubernetes readiness probe endpoint.

    This endpoint checks if the application is ready to serve traffic.
    It verifies that critical services (Ollama, Whisper) are available.

    Returns:
    - HTTP 200 if ready to serve traffic
    - HTTP 503 if not ready (dependencies unavailable)
    """
)
async def readiness_probe(
    pipeline: MedicalDocumentationPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings)
) -> ReadinessResponse:
    """
    Check if the application is ready to serve traffic.

    Readiness means:
    - Pipeline is loaded
    - Ollama is accessible
    - Whisper model can be loaded

    Args:
        pipeline: Injected pipeline instance
        settings: Injected application settings

    Returns:
        ReadinessResponse with ready status

    Raises:
        HTTPException: 503 if not ready
    """
    try:
        # Check pipeline is loaded
        if pipeline is None:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline not initialized"
            )

        # Quick check for Ollama (don't wait for full test)
        ollama_result = check_ollama(pipeline, settings)
        if ollama_result.status == ServiceStatus.UNHEALTHY:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ollama not ready: {ollama_result.message}"
            )

        logger.debug("Readiness probe: READY")
        return ReadinessResponse(
            status="ready",
            message="Application is ready to serve traffic",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    except Exception as e:
        logger.warning(f"Readiness probe failed: {e}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Not ready: {str(e)}"
        )


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Kubernetes liveness probe",
    description="""
    Kubernetes liveness probe endpoint.

    This endpoint performs a minimal check to verify the application process is alive.
    It does not check external dependencies.

    Returns:
    - HTTP 200 if the application process is running
    """
)
async def liveness_probe() -> LivenessResponse:
    """
    Check if the application process is alive.

    This is a lightweight endpoint that simply confirms the API can respond.
    It does not check external dependencies - that's what readiness is for.

    Returns:
        LivenessResponse with alive status
    """
    logger.debug("Liveness probe: ALIVE")
    return LivenessResponse(
        status="alive",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
