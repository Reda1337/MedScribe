"""
Configuration Management for DocuMed AI
========================================

This module handles all application configuration using the Settings pattern
with Pydantic. This approach provides:

1. **Environment Variable Support**: Easy deployment configuration
2. **Validation**: Catches configuration errors at startup
3. **Type Safety**: IDE support and runtime validation
4. **Defaults**: Sensible defaults for development
5. **Documentation**: Self-documenting configuration

Design Pattern: Singleton-like Settings
We use a cached function to ensure we only load settings once,
but still allow for easy testing with different configurations.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Environment variables are prefixed with DOCUMED_ to avoid conflicts.
    Example: DOCUMED_OLLAMA_MODEL=llama3.2
    
    Priority order (highest to lowest):
    1. Environment variables
    2. .env file
    3. Default values defined here
    """
    
    # =================================================================
    # Whisper Configuration
    # =================================================================
    whisper_model: str = Field(
        default="base",
        description="""
        Whisper model size. Options: tiny, base, small, medium, large
        
        Trade-offs:
        - tiny: Fastest, least accurate, ~1GB VRAM
        - base: Good balance for development, ~1GB VRAM  
        - small: Better accuracy, ~2GB VRAM
        - medium: High accuracy, ~5GB VRAM
        - large: Best accuracy, ~10GB VRAM
        
        For medical transcription, recommend 'small' or higher in production.
        """
    )
    
    whisper_device: str = Field(
        default="cpu",
        description="Device for Whisper: 'cpu', 'cuda', or 'auto'"
    )
    
    whisper_language: Optional[str] = Field(
        default=None,
        description="Force language detection. None = auto-detect"
    )

    # =================================================================
    # Speaker Diarization Configuration (Phase 1)
    # =================================================================
    enable_diarization: bool = Field(
        default=True,
        description="""
        Enable speaker diarization to identify 'who spoke when'.

        HIGHLY RECOMMENDED for medical consultations to prevent LLM
        hallucinations where doctor is misidentified as having symptoms.
        """
    )

    diarization_model: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="Pyannote model for speaker diarization"
    )

    diarization_min_speakers: Optional[int] = Field(
        default=2,
        description="Minimum number of speakers (None = auto-detect)"
    )

    diarization_max_speakers: Optional[int] = Field(
        default=2,
        description="Maximum number of speakers (None = auto-detect)"
    )

    diarization_device: str = Field(
        default="cpu",
        description="Device for diarization: 'cpu' or 'cuda'"
    )

    huggingface_token: Optional[str] = Field(
        default=None,
        description="""
        HuggingFace authentication token for Pyannote models.
        Get token at: https://huggingface.co/settings/tokens
        Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1

        Can also be set via HUGGINGFACE_TOKEN environment variable.
        """
    )

    auto_label_speakers: bool = Field(
        default=True,
        description="""
        Automatically label speakers as Doctor/Patient based on speaking time.

        Heuristic: Speaker who talks more = Doctor
        Disable if you want to manually configure speaker roles.
        """
    )

    # =================================================================
    # Ollama Configuration
    # =================================================================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL. Default is local installation."
    )
    
    ollama_model: str = Field(
        default="llama3.2",
        description="""
        Ollama model for SOAP generation. Recommended models:
        
        - llama3.2: Good balance of speed and quality (8B params)
        - llama3.2:70b: Higher quality, needs more resources
        - mistral: Fast, good for structured output
        - mixtral: High quality, slower
        
        For medical use, larger models generally perform better with
        medical terminology and reasoning.
        """
    )
    
    ollama_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="""
        Temperature for text generation (0.0 - 2.0)
        
        - 0.0-0.3: More deterministic, consistent output (recommended for medical)
        - 0.4-0.7: Balanced creativity and consistency
        - 0.8+: More creative, less predictable
        
        For medical documentation, lower is better for consistency.
        """
    )
    
    ollama_timeout: int = Field(
        default=120,
        description="Timeout in seconds for Ollama requests"
    )

    ollama_context_window: int = Field(
        default=4096,
        description="Context window size for Ollama model (tokens)"
    )

    # =================================================================
    # Processing Configuration
    # =================================================================
    max_audio_duration_seconds: int = Field(
        default=1800,  # 30 minutes
        description="Maximum audio duration to process (prevents resource exhaustion)"
    )
    
    supported_audio_formats: list[str] = Field(
        default=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
        description="List of supported audio file extensions"
    )

    # =================================================================
    # Context Window Management (Phase 5 - Map-Reduce)
    # =================================================================
    enable_map_reduce: bool = Field(
        default=True,
        description="""
        Enable Map-Reduce strategy for long transcripts.

        When transcript exceeds context window, split into chunks,
        summarize each, then synthesize final SOAP note.
        """
    )

    map_reduce_chunk_size: int = Field(
        default=3000,
        description="""
        Chunk size in tokens for Map-Reduce strategy.

        Should be smaller than context window to allow for prompt overhead.
        """
    )

    map_reduce_overlap: int = Field(
        default=200,
        description="Token overlap between chunks to preserve context"
    )

    # =================================================================
    # Output Configuration
    # =================================================================
    output_dir: str = Field(
        default="./output",
        description="Directory for output files"
    )
    
    save_transcriptions: bool = Field(
        default=True,
        description="Whether to save intermediate transcriptions"
    )

    # =================================================================
    # HIPAA Compliance & Security (Phase 6)
    # =================================================================
    enable_encryption: bool = Field(
        default=True,
        description="""
        Enable AES-256 encryption for all saved audio and transcripts.

        REQUIRED for HIPAA compliance when storing PHI (Protected Health Information).
        2025 HIPAA mandates encryption at rest (no longer "addressable").
        """
    )

    encryption_key_path: Optional[str] = Field(
        default=None,
        description="""
        Path to encryption key file.

        If None, generates ephemeral key (lost after process exit).
        For production, use persistent key stored securely (HSM recommended).
        """
    )

    enable_audit_logging: bool = Field(
        default=True,
        description="""
        Enable HIPAA-compliant audit logging.

        Logs: user, timestamp, file accessed, action taken.
        Required for HIPAA compliance (45 CFR § 164.312(b)).
        """
    )

    audit_log_path: str = Field(
        default="./audit.log",
        description="Path to audit log file (encrypted)"
    )

    secure_temp_directory: bool = Field(
        default=True,
        description="""
        Use encrypted temporary directory for processing.

        Ensures no plaintext PHI in temp files.
        Automatic cleanup after processing.
        """
    )

    # =================================================================
    # Clinical Coding (Phase 2)
    # =================================================================
    enable_clinical_coding: bool = Field(
        default=True,
        description="""
        Enable ICD-10 and CPT code suggestions in SOAP notes.

        Helps with billing and medical terminology standardization.
        """
    )

    coding_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="""
        Minimum confidence score to include AI-suggested codes.

        Codes below this threshold are omitted (reduce false positives).
        """
    )

    # =================================================================
    # Logging Configuration
    # =================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Python logging format string"
    )
    
    class Config:
        """Pydantic configuration for Settings."""
        env_prefix = "DOCUMED_"  # All env vars start with DOCUMED_
        env_file = ".env"  # Load from .env file if present
        env_file_encoding = "utf-8"
        case_sensitive = False  # DOCUMED_WHISPER_MODEL = documed_whisper_model


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached singleton).
    
    Using lru_cache ensures we only parse environment variables once.
    This is important because:
    1. Parsing is relatively slow
    2. Settings should be consistent throughout app lifecycle
    3. Reduces memory usage
    
    For testing, you can clear the cache:
        get_settings.cache_clear()
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


def get_settings_for_testing(**overrides) -> Settings:
    """
    Create a Settings instance with custom values for testing.
    
    This factory function allows tests to easily create Settings
    with specific values without affecting the global settings.
    
    Example:
        settings = get_settings_for_testing(
            whisper_model="tiny",
            ollama_model="llama3.2"
        )
    
    Args:
        **overrides: Setting values to override
        
    Returns:
        Settings: New Settings instance with overrides applied
    """
    return Settings(**overrides)
