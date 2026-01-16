"""
Custom Exceptions for DocuMed AI
================================

This module defines a hierarchy of custom exceptions that:

1. **Categorize Errors**: Different exception types for different problems
2. **Carry Context**: Include relevant information for debugging
3. **Enable Recovery**: Allow calling code to handle specific errors
4. **Support APIs**: Map cleanly to HTTP status codes (future)

Exception Hierarchy:
    DocuMedError (base)
    ├── AudioError
    │   ├── AudioFileNotFoundError
    │   ├── UnsupportedAudioFormatError
    │   └── AudioTooLongError
    ├── TranscriptionError
    │   └── WhisperModelError
    ├── GenerationError
    │   └── OllamaConnectionError
    └── ConfigurationError
"""

from typing import Optional


class DocuMedError(Exception):
    """
    Base exception for all DocuMed errors.
    
    All custom exceptions inherit from this, allowing code to catch
    all DocuMed-related errors with a single except clause:
    
        try:
            process_audio(file)
        except DocuMedError as e:
            logger.error(f"DocuMed error: {e}")
    
    Attributes:
        message: Human-readable error description
        details: Additional context (dict for API responses)
    """
    
    def __init__(
        self, 
        message: str, 
        details: Optional[dict] = None
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for API responses.
        
        Returns a structured error that can be easily serialized to JSON.
        This is crucial for the future API layer.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


# =============================================================================
# Audio-Related Errors
# =============================================================================

class AudioError(DocuMedError):
    """Base class for audio-related errors."""
    pass


class AudioFileNotFoundError(AudioError):
    """Raised when the specified audio file doesn't exist."""
    
    def __init__(self, file_path: str):
        super().__init__(
            message=f"Audio file not found: {file_path}",
            details={"file_path": file_path}
        )


class UnsupportedAudioFormatError(AudioError):
    """Raised when the audio file format is not supported."""
    
    def __init__(self, file_path: str, format: str, supported_formats: list[str]):
        super().__init__(
            message=f"Unsupported audio format: {format}. Supported: {', '.join(supported_formats)}",
            details={
                "file_path": file_path,
                "format": format,
                "supported_formats": supported_formats
            }
        )


class AudioTooLongError(AudioError):
    """Raised when audio exceeds maximum allowed duration."""
    
    def __init__(
        self, 
        file_path: str, 
        duration: float, 
        max_duration: float
    ):
        super().__init__(
            message=f"Audio too long: {duration:.1f}s (max: {max_duration:.1f}s)",
            details={
                "file_path": file_path,
                "duration_seconds": duration,
                "max_duration_seconds": max_duration
            }
        )


# =============================================================================
# Transcription-Related Errors
# =============================================================================

class TranscriptionError(DocuMedError):
    """Base class for transcription errors."""
    pass


class WhisperModelError(TranscriptionError):
    """Raised when there's an issue with the Whisper model."""
    
    def __init__(self, model_name: str, original_error: str):
        super().__init__(
            message=f"Whisper model error ({model_name}): {original_error}",
            details={
                "model_name": model_name,
                "original_error": original_error
            }
        )


class TranscriptionFailedError(TranscriptionError):
    """Raised when transcription fails for any reason."""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message=f"Transcription failed for {file_path}: {reason}",
            details={
                "file_path": file_path,
                "reason": reason
            }
        )


# =============================================================================
# Generation-Related Errors
# =============================================================================

class GenerationError(DocuMedError):
    """Base class for SOAP generation errors."""
    pass


class OllamaConnectionError(GenerationError):
    """Raised when we can't connect to Ollama."""
    
    def __init__(self, url: str, original_error: str):
        super().__init__(
            message=f"Cannot connect to Ollama at {url}: {original_error}",
            details={
                "ollama_url": url,
                "original_error": original_error,
                "hint": "Make sure Ollama is running: 'ollama serve'"
            }
        )


class ModelNotFoundError(GenerationError):
    """Raised when the requested Ollama model is not available."""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found in Ollama",
            details={
                "model_name": model_name,
                "hint": f"Pull the model first: 'ollama pull {model_name}'"
            }
        )


class SOAPGenerationError(GenerationError):
    """Raised when SOAP note generation fails."""
    
    def __init__(self, reason: str, transcription_preview: str = ""):
        preview = transcription_preview[:100] + "..." if len(transcription_preview) > 100 else transcription_preview
        super().__init__(
            message=f"Failed to generate SOAP note: {reason}",
            details={
                "reason": reason,
                "transcription_preview": preview
            }
        )


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(DocuMedError):
    """Raised when there's a configuration problem."""
    
    def __init__(self, setting_name: str, issue: str):
        super().__init__(
            message=f"Configuration error for '{setting_name}': {issue}",
            details={
                "setting_name": setting_name,
                "issue": issue
            }
        )
