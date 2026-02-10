"""
API Response Models
===================

Pydantic models for API responses.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration for tracking processing states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Response model for job submission endpoints."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "message": "Job submitted successfully",
                "created_at": "2024-01-17T10:30:00Z"
            }
        }
    )

    job_id: str = Field(
        ...,
        description="Unique identifier for the submitted job"
    )
    status: JobStatus = Field(
        ...,
        description="Current status of the job"
    )
    message: str = Field(
        ...,
        description="Human-readable message about the job submission"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the job was created"
    )

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from ISO format string if needed."""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v


class JobStatusResponse(BaseModel):
    """Response model for job status check endpoints."""

    model_config = ConfigDict(
        extra='ignore',
        json_schema_extra={
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "progress": 45,
                "current_stage": "Generating SOAP note",
                "result": None,
                "error": None,
                "created_at": "2024-01-17T10:30:00Z",
                "updated_at": "2024-01-17T10:31:30Z"
            }
        }
    )

    job_id: str = Field(
        ...,
        description="Unique identifier for the job"
    )
    status: JobStatus = Field(
        ...,
        description="Current status of the job"
    )
    progress: Optional[int] = Field(
        default=None,
        description="Progress percentage (0-100)",
        ge=0,
        le=100
    )
    current_stage: Optional[str] = Field(
        default=None,
        description="Current processing stage description"
    )
    result: Optional[Any] = Field(
        default=None,
        description="Job result (available when status is 'completed')"
    )
    error: Optional[dict] = Field(
        default=None,
        description="Error details (available when status is 'failed')"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the job was created"
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp when the job was last updated"
    )

    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime from ISO format string if needed."""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v
