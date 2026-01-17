"""
API Request Models
==================

Pydantic models for API request validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ProcessRequest(BaseModel):
    """Request model for full pipeline processing (audio → transcription → SOAP note)."""

    language: str = Field(
        default="en",
        description="Language code for transcription (en, es, fr, etc.)",
        examples=["en", "es", "fr"]
    )
    generate_audio: bool = Field(
        default=False,
        description="Whether to generate audio output for the SOAP note"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "language": "en",
                "generate_audio": False
            }
        }


class TranscribeRequest(BaseModel):
    """Request model for transcription-only processing."""

    language: str = Field(
        default="en",
        description="Language code for transcription (en, es, fr, etc.)",
        examples=["en", "es", "fr"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "language": "en"
            }
        }


class GenerateSOAPRequest(BaseModel):
    """Request model for SOAP note generation from existing transcription text."""

    transcription: str = Field(
        ...,
        description="The transcribed text from the medical conversation",
        min_length=1
    )
    language: str = Field(
        default="en",
        description="Language code for SOAP generation (en, es, fr, etc.)",
        examples=["en", "es", "fr"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "transcription": "Patient presents with fever and cough for 3 days...",
                "language": "en"
            }
        }
