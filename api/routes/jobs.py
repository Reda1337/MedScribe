"""
Job Management Endpoints
========================

API endpoints for submitting and tracking background processing jobs.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from typing import Optional
from datetime import datetime

from api.dependencies import get_job_manager, get_current_user_optional
from api.models.requests import GenerateSOAPRequest
from api.models.responses import JobResponse, JobStatusResponse, JobStatus
from api.utils.file_handler import FileHandler
from api.services.job_manager import JobManager
from tasks import process_audio_task, transcribe_audio_task, generate_soap_task

router = APIRouter()


@router.post("/process", response_model=JobResponse)
async def submit_process_job(
    file: UploadFile = File(..., description="Audio file to process"),
    user: Optional[dict] = Depends(get_current_user_optional),
    job_manager: JobManager = Depends(get_job_manager)
):
    """
    Submit a full pipeline processing job (audio → transcription → SOAP note).

    Upload an audio file and receive a job ID for tracking progress.

    Args:
        file: Audio file (mp3, wav, m4a, flac)
        user: Optional authenticated user
        job_manager: Job manager instance

    Returns:
        JobResponse with job_id and initial status
    """
    file_handler = FileHandler()

    # Create job
    job_id = job_manager.create_job(
        job_type="process",
        metadata={
            "filename": file.filename,
            "user_id": user.get("user_id") if user else None
        }
    )

    # Save uploaded file
    temp_path = await file_handler.save_upload_file(file, job_id)

    # Submit to Celery
    process_audio_task.delay(job_id, str(temp_path))

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job submitted successfully. Processing will begin shortly.",
        created_at=datetime.utcnow()
    )


@router.post("/transcribe", response_model=JobResponse)
async def submit_transcribe_job(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    user: Optional[dict] = Depends(get_current_user_optional),
    job_manager: JobManager = Depends(get_job_manager)
):
    """
    Submit a transcription-only job (no SOAP note generation).

    Args:
        file: Audio file (mp3, wav, m4a, flac)
        user: Optional authenticated user
        job_manager: Job manager instance

    Returns:
        JobResponse with job_id and initial status
    """
    file_handler = FileHandler()

    job_id = job_manager.create_job(
        job_type="transcribe",
        metadata={
            "filename": file.filename,
            "user_id": user.get("user_id") if user else None
        }
    )

    temp_path = await file_handler.save_upload_file(file, job_id)
    transcribe_audio_task.delay(job_id, str(temp_path))

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Transcription job submitted successfully.",
        created_at=datetime.utcnow()
    )


@router.post("/generate-soap", response_model=JobResponse)
async def submit_soap_generation_job(
    request: GenerateSOAPRequest,
    user: Optional[dict] = Depends(get_current_user_optional),
    job_manager: JobManager = Depends(get_job_manager)
):
    """
    Submit a SOAP generation job from existing transcription text.

    No file upload required - provide transcription text directly.

    Args:
        request: Request containing transcription text and language
        user: Optional authenticated user
        job_manager: Job manager instance

    Returns:
        JobResponse with job_id and initial status
    """
    job_id = job_manager.create_job(
        job_type="generate_soap",
        metadata={
            "language": request.language,
            "user_id": user.get("user_id") if user else None,
            "transcription_length": len(request.transcription)
        }
    )

    generate_soap_task.delay(job_id, request.transcription, request.language)

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="SOAP generation job submitted successfully.",
        created_at=datetime.utcnow()
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager)
):
    """
    Get the current status and progress of a job.

    Args:
        job_id: Unique job identifier
        job_manager: Job manager instance

    Returns:
        JobStatusResponse with current status, progress, and result (if completed)

    Raises:
        404: Job not found
    """
    job_data = job_manager.get_job(job_id)

    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found. It may have expired or never existed."
        )

    return JobStatusResponse(**job_data)


@router.get("/jobs/{job_id}/result")
async def get_job_result(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager)
):
    """
    Get the result of a completed job.

    This endpoint returns only the result payload without status metadata.

    Args:
        job_id: Unique job identifier
        job_manager: Job manager instance

    Returns:
        The job result (transcription and/or SOAP note)

    Raises:
        404: Job not found
        400: Job not completed yet
    """
    job_data = job_manager.get_job(job_id)

    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found. It may have expired or never existed."
        )

    if job_data["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed yet. Current status: {job_data['status']}"
        )

    return job_data["result"]
