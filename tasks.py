"""
Celery Tasks for MedScribe AI Background Processing
=================================================

This module contains Celery tasks for handling async background jobs:
- Audio transcription
- SOAP note generation
- Full pipeline processing

Tasks are executed by Celery workers and tracked via Redis.
"""

from celery import Celery
from pathlib import Path
import asyncio

from config import get_settings
from pipeline import create_pipeline
from api.services.job_manager import JobManager
from api.utils.file_handler import FileHandler
from models import ProcessingStatus

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "MedScribe",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


def progress_callback(job_id: str):
    """
    Create progress callback for pipeline.

    Args:
        job_id: Job identifier

    Returns:
        Callback function that updates job progress
    """
    job_manager = JobManager()

    def callback(status: ProcessingStatus, message: str):
        # Map ProcessingStatus to progress percentage
        progress_map = {
            ProcessingStatus.PENDING: 0,
            ProcessingStatus.TRANSCRIBING: 25,
            ProcessingStatus.GENERATING: 75,
            ProcessingStatus.COMPLETED: 100,
            ProcessingStatus.FAILED: 0
        }

        progress = progress_map.get(status, 0)
        job_manager.set_job_progress(job_id, progress, status.value)

    return callback


@celery_app.task(name="tasks.process_audio")
def process_audio_task(job_id: str, audio_path: str):
    """
    Celery task for full pipeline processing.

    Args:
        job_id: Job identifier
        audio_path: Path to audio file

    Returns:
        Result dictionary with transcription and SOAP note
    """
    job_manager = JobManager()
    file_handler = FileHandler()

    try:
        # Create pipeline
        pipeline = create_pipeline()

        # Run async processing in sync context
        result = asyncio.run(
            pipeline.aprocess(
                audio_path=audio_path,
                progress_callback=progress_callback(job_id)
            )
        )

        # Convert result to dict (with JSON serialization for datetime objects)
        result_dict = result.model_dump(mode='json')

        # Mark as completed
        job_manager.set_job_completed(job_id, result_dict)

        # Cleanup temp file
        file_handler.cleanup_file(audio_path)

        return result_dict

    except Exception as e:
        # Mark as failed
        job_manager.set_job_failed(job_id, {
            "error": type(e).__name__,
            "message": str(e)
        })

        # Cleanup temp file
        file_handler.cleanup_file(audio_path)

        raise


@celery_app.task(name="tasks.transcribe_audio")
def transcribe_audio_task(job_id: str, audio_path: str):
    """
    Celery task for transcription only.

    Args:
        job_id: Job identifier
        audio_path: Path to audio file

    Returns:
        Result dictionary with transcription
    """
    job_manager = JobManager()
    file_handler = FileHandler()

    try:
        # Update progress: starting transcription
        job_manager.set_job_progress(job_id, 10, "transcribing")

        pipeline = create_pipeline()

        result = asyncio.run(
            pipeline.atranscribe_only(
                audio_path=audio_path
            )
        )

        # Update progress: transcription complete
        job_manager.set_job_progress(job_id, 90, "finalizing")

        result_dict = result.model_dump(mode='json')
        job_manager.set_job_completed(job_id, result_dict)
        file_handler.cleanup_file(audio_path)

        return result_dict

    except Exception as e:
        job_manager.set_job_failed(job_id, {
            "error": type(e).__name__,
            "message": str(e)
        })
        file_handler.cleanup_file(audio_path)
        raise


@celery_app.task(name="tasks.generate_soap")
def generate_soap_task(job_id: str, transcription: str, language: str = "en"):
    """
    Celery task for SOAP generation only.

    Args:
        job_id: Job identifier
        transcription: Transcribed text
        language: Language code

    Returns:
        Result dictionary with SOAP note
    """
    job_manager = JobManager()

    try:
        # Update progress: starting SOAP generation
        job_manager.set_job_progress(job_id, 10, "generating")

        pipeline = create_pipeline()

        result = asyncio.run(
            pipeline.agenerate_soap_only(
                transcription=transcription,
                language=language
            )
        )

        # Update progress: generation complete
        job_manager.set_job_progress(job_id, 90, "finalizing")

        result_dict = result.model_dump(mode='json')
        job_manager.set_job_completed(job_id, result_dict)

        return result_dict

    except Exception as e:
        job_manager.set_job_failed(job_id, {
            "error": type(e).__name__,
            "message": str(e)
        })
        raise
