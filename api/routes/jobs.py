"""
Job Management Endpoints
========================

API endpoints for submitting and tracking background processing jobs.
Will be implemented in Phase 5.
"""

from fastapi import APIRouter

router = APIRouter()

# TODO: Implement in Phase 5
# - POST /process - Submit full pipeline job
# - POST /transcribe - Submit transcription-only job
# - POST /generate-soap - Submit SOAP generation job
# - GET /jobs/{job_id} - Get job status
# - GET /jobs/{job_id}/result - Get job result
