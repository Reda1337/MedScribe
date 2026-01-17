"""
File Upload Handler
===================

Async file upload handling with validation and cleanup.
"""

import aiofiles
import os
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException, status

from config import get_settings
from exceptions import UnsupportedAudioFormatError


class FileHandler:
    """Handle file uploads and temporary file management."""

    def __init__(self):
        """Initialize FileHandler with settings from config."""
        self.settings = get_settings()
        self.temp_dir = Path(self.settings.temp_file_dir or tempfile.gettempdir())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        self.allowed_formats = self.settings.allowed_audio_formats

    async def save_upload_file(self, upload_file: UploadFile, job_id: str) -> Path:
        """
        Save uploaded file to temporary location with validation.

        Args:
            upload_file: FastAPI UploadFile object from request
            job_id: Unique job ID for naming the file

        Returns:
            Path to the saved file

        Raises:
            UnsupportedAudioFormatError: If file format is not allowed
            HTTPException: If file size exceeds limit
        """
        # Validate file extension
        if not upload_file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )

        file_ext = Path(upload_file.filename).suffix.lower()
        if file_ext not in self.allowed_formats:
            raise UnsupportedAudioFormatError(
                f"File format '{file_ext}' not supported. "
                f"Allowed formats: {', '.join(self.allowed_formats)}"
            )

        # Create temp file path with job ID and original extension
        temp_file_path = self.temp_dir / f"{job_id}{file_ext}"

        # Save file with size validation (chunked upload)
        total_size = 0
        chunk_size = 8192  # 8KB chunks

        try:
            async with aiofiles.open(temp_file_path, 'wb') as f:
                while True:
                    chunk = await upload_file.read(chunk_size)
                    if not chunk:
                        break

                    total_size += len(chunk)

                    # Check file size limit
                    if total_size > self.max_size_bytes:
                        # Clean up partial file
                        await f.close()
                        temp_file_path.unlink(missing_ok=True)

                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File too large. Maximum size: {self.settings.max_upload_size_mb}MB"
                        )

                    await f.write(chunk)

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Clean up on any error
            temp_file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving file: {str(e)}"
            )

        return temp_file_path

    def cleanup_file(self, file_path: Path | str) -> None:
        """
        Delete temporary file.

        Args:
            file_path: Path to file to delete (can be Path or string)

        Note:
            This is a best-effort cleanup. Errors are logged but not raised
            to prevent cleanup failures from affecting job completion.
        """
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            # Log but don't raise - cleanup is best-effort
            print(f"Warning: Could not delete temp file {file_path}: {e}")

    def get_file_size(self, file_path: Path | str) -> int:
        """
        Get file size in bytes.

        Args:
            file_path: Path to file

        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size

    def get_file_size_mb(self, file_path: Path | str) -> float:
        """
        Get file size in megabytes.

        Args:
            file_path: Path to file

        Returns:
            File size in MB (rounded to 2 decimal places)
        """
        size_bytes = self.get_file_size(file_path)
        return round(size_bytes / (1024 * 1024), 2)
