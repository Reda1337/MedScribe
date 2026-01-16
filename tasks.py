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


# TODO: Implement Celery tasks in Phase 5
# - process_audio_task: Full pipeline processing
# - transcribe_audio_task: Transcription only
# - generate_soap_task: SOAP generation only
# See Phase 5 in the implementation plan for details
