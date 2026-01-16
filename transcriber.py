"""
Transcription Service for DocuMed AI
====================================

This module handles audio-to-text conversion using OpenAI's Whisper model.

Architecture Pattern: Protocol-based Service
--------------------------------------------
We use Python's Protocol (similar to interfaces in other languages) to define
what a Transcriber should do. This allows:
1. Easy swapping of implementations (local Whisper vs API)
2. Simple mocking for tests
3. Clear contracts for service behavior

Why Whisper?
------------
- Open source and free
- Excellent accuracy, especially for medical terminology
- Works offline (privacy-friendly for medical data)
- Multiple model sizes for different hardware constraints
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Protocol

# We'll use openai-whisper for local transcription
import whisper
import numpy as np

from config import Settings, get_settings
from models import TranscriptionResult, SpeakerSegment, DiarizationResult
from exceptions import (
    AudioFileNotFoundError,
    UnsupportedAudioFormatError,
    AudioTooLongError,
    TranscriptionFailedError,
    WhisperModelError,
)

# Phase 1: Import speaker diarization
from speaker_diarizer import (
    create_speaker_diarizer,
    SpeakerDiarizerProtocol,
    merge_diarization_with_transcription,
)


# Set up module logger
logger = logging.getLogger(__name__)


class TranscriberProtocol(Protocol):
    """
    Protocol defining the interface for transcription services.
    
    Using Protocol instead of ABC because:
    1. Protocols support structural subtyping (duck typing with types)
    2. Implementations don't need to explicitly inherit
    3. Better for dependency injection patterns
    
    Any class with a `transcribe` method matching this signature
    will be considered a valid TranscriberProtocol.
    """
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file to text (synchronous).
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            TranscriptionResult containing the transcribed text
            
        Raises:
            AudioFileNotFoundError: If file doesn't exist
            UnsupportedAudioFormatError: If format not supported
            TranscriptionFailedError: If transcription fails
        """
        ...
    
    async def atranscribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file to text (asynchronous).
        
        This is the async version for use with FastAPI, aiohttp, etc.
        Uses thread pool to avoid blocking the event loop.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            TranscriptionResult containing the transcribed text
        """
        ...


class WhisperTranscriber:
    """
    Transcriber implementation using local Whisper model.

    Enhanced with Phase 1: Speaker diarization support to identify
    "who spoke when" in medical consultations. This prevents LLM
    hallucinations where the doctor is misidentified as having symptoms.

    This is the primary implementation for our medical transcription needs.
    Using local Whisper ensures:
    1. Data privacy (no audio sent to cloud)
    2. No API costs
    3. Works offline
    4. Full control over model behavior

    The class follows the Single Responsibility Principle - it only
    handles transcription, nothing else.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[whisper.Whisper] = None,
        speaker_diarizer: Optional[SpeakerDiarizerProtocol] = None
    ):
        """
        Initialize the Whisper transcriber.

        Args:
            settings: Application settings (uses defaults if not provided)
            model: Pre-loaded Whisper model (loads fresh if not provided)
            speaker_diarizer: Speaker diarization service (auto-created if not provided)

        Dependency Injection Pattern:
        - Settings and model can be injected for testing
        - If not provided, uses defaults (good for production)
        """
        self.settings = settings or get_settings()
        self._model = model
        self._model_loaded = model is not None

        # Phase 1: Speaker diarization support
        self._speaker_diarizer = speaker_diarizer
        self._diarizer_initialized = speaker_diarizer is not None

        logger.info(
            f"WhisperTranscriber initialized with model: {self.settings.whisper_model}, "
            f"diarization: {self.settings.enable_diarization}"
        )
    
    @property
    def model(self) -> whisper.Whisper:
        """
        Lazy-load the Whisper model.

        Property Pattern Benefits:
        1. Model only loaded when first needed (faster startup)
        2. Same model instance reused for all transcriptions
        3. Transparent to calling code
        """
        if not self._model_loaded:
            self._load_model()
        return self._model

    @property
    def speaker_diarizer(self) -> Optional[SpeakerDiarizerProtocol]:
        """
        Lazy-load the speaker diarizer.

        Only initialized if diarization is enabled in settings.
        """
        if not self.settings.enable_diarization:
            return None

        if not self._diarizer_initialized:
            self._init_speaker_diarizer()

        return self._speaker_diarizer

    def _init_speaker_diarizer(self) -> None:
        """Initialize the speaker diarization service."""
        try:
            logger.info("Initializing speaker diarization service")

            self._speaker_diarizer = create_speaker_diarizer(
                use_mock=False,
                model_name=self.settings.diarization_model,
                min_speakers=self.settings.diarization_min_speakers,
                max_speakers=self.settings.diarization_max_speakers,
                auth_token=self.settings.huggingface_token,
                device=self.settings.diarization_device,
            )

            self._diarizer_initialized = True
            logger.info("Speaker diarization service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize speaker diarization: {e}")
            logger.warning("Continuing without speaker diarization")
            self._speaker_diarizer = None
            self._diarizer_initialized = True  # Mark as initialized to prevent retry
    
    def _load_model(self) -> None:
        """
        Load the Whisper model into memory.
        
        This is separate from __init__ for:
        1. Lazy loading (don't load until needed)
        2. Better error handling
        3. Easier testing (can mock before load)
        """
        try:
            logger.info(f"Loading Whisper model: {self.settings.whisper_model}")
            
            self._model = whisper.load_model(
                self.settings.whisper_model,
                device=self.settings.whisper_device
            )
            self._model_loaded = True
            
            logger.info(
                f"Whisper model loaded successfully on {self.settings.whisper_device}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise WhisperModelError(
                model_name=self.settings.whisper_model,
                original_error=str(e)
            )
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Enhanced with Phase 1: Speaker diarization support.

        This is the main public method. It:
        1. Validates the input file
        2. (Optional) Runs speaker diarization
        3. Runs transcription
        4. Merges diarization with transcription
        5. Returns structured result with speaker labels

        Args:
            audio_path: Path to the audio file

        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        # Step 1: Validate input
        self._validate_audio_file(audio_path)

        logger.info(f"Starting transcription of: {audio_path}")

        try:
            # Step 2 (Phase 1): Run speaker diarization if enabled
            diarization_result = None
            if self.speaker_diarizer is not None:
                try:
                    logger.info("Running speaker diarization...")
                    diarization_result = self.speaker_diarizer.diarize(audio_path)

                    # Apply speaker role labels (Doctor/Patient)
                    if self.settings.auto_label_speakers:
                        diarization_result = self.speaker_diarizer.apply_labels_to_segments(
                            diarization_result,
                            apply_labels=True
                        )

                    logger.info(
                        f"Diarization complete: {diarization_result.num_speakers} speakers, "
                        f"{len(diarization_result.segments)} segments"
                    )
                except Exception as e:
                    logger.error(f"Speaker diarization failed: {e}")
                    logger.warning("Continuing with transcription without speaker labels")
                    diarization_result = None

            # Step 3: Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=self.settings.whisper_language,
                verbose=False,  # Suppress Whisper's output
                word_timestamps=True  # Enable word-level timestamps for better diarization merge
            )

            # Step 4: Extract results
            text = result["text"].strip()
            language = result.get("language", "en")

            # Calculate duration from segments if available
            duration = 0.0
            if result.get("segments"):
                duration = result["segments"][-1].get("end", 0.0)

            # Step 5 (Phase 1): Merge diarization with transcription
            if diarization_result is not None:
                logger.info("Merging transcription with speaker diarization...")
                word_timestamps = []

                # Extract word-level timestamps from Whisper result
                if result.get("segments"):
                    for segment in result["segments"]:
                        if segment.get("words"):
                            word_timestamps.extend(segment["words"])

                # Merge diarization with transcription
                diarization_result = merge_diarization_with_transcription(
                    diarization_result,
                    text,
                    word_timestamps if word_timestamps else None
                )

                # Create formatted transcript with speaker labels
                formatted_text = diarization_result.get_formatted_transcript()

                logger.info(
                    f"Transcription complete: {len(text)} characters, "
                    f"{duration:.1f}s duration, language: {language}, "
                    f"with {diarization_result.num_speakers} speakers"
                )

                return TranscriptionResult(
                    text=formatted_text,  # Use speaker-labeled text
                    language=language,
                    duration_seconds=duration,
                    confidence=None,  # Whisper doesn't provide overall confidence
                    speaker_segments=diarization_result.segments,
                    diarization=diarization_result
                )
            else:
                # No diarization - return standard result
                logger.info(
                    f"Transcription complete: {len(text)} characters, "
                    f"{duration:.1f}s duration, language: {language}"
                )

                return TranscriptionResult(
                    text=text,
                    language=language,
                    duration_seconds=duration,
                    confidence=None  # Whisper doesn't provide overall confidence
                )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise TranscriptionFailedError(
                file_path=audio_path,
                reason=str(e)
            )

    async def atranscribe(self, audio_path: str) -> TranscriptionResult:
        """
        Async version of transcribe() for use with FastAPI/async frameworks.

        Since Whisper is a CPU/GPU-bound blocking operation, we use
        asyncio.to_thread() to run it in a thread pool. This prevents
        blocking the event loop while still getting async benefits.

        Thread Pool Strategy:
        - The transcription runs in a separate thread
        - The event loop remains responsive to other requests
        - Other I/O operations (DB, network) can proceed concurrently

        Args:
            audio_path: Path to the audio file

        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        logger.info(f"Starting async transcription of: {audio_path}")

        # Run the blocking transcribe() method in a thread pool
        # This is the recommended pattern for CPU-bound work in async contexts
        result = await asyncio.to_thread(self.transcribe, audio_path)

        logger.info(f"Async transcription complete: {len(result.text)} chars")
        return result
    
    def _validate_audio_file(self, audio_path: str) -> None:
        """
        Validate that the audio file exists and is supported.
        
        Validation is separate from transcription for:
        1. Early failure (fail fast principle)
        2. Clear error messages
        3. Testability
        """
        path = Path(audio_path)
        
        # Check file exists
        if not path.exists():
            raise AudioFileNotFoundError(audio_path)
        
        # Check format is supported
        extension = path.suffix.lower().lstrip('.')
        if extension not in self.settings.supported_audio_formats:
            raise UnsupportedAudioFormatError(
                file_path=audio_path,
                format=extension,
                supported_formats=self.settings.supported_audio_formats
            )
        
        # Check file size (basic sanity check)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:  # 500MB limit
            logger.warning(f"Large audio file: {file_size_mb:.1f}MB")
        
        logger.debug(f"Audio file validated: {audio_path}")


class MockTranscriber:
    """
    Mock transcriber for testing.
    
    This demonstrates the value of our Protocol pattern - we can
    easily create alternative implementations for different scenarios.
    Supports both sync and async interfaces for comprehensive testing.
    
    Usage in tests:
        transcriber = MockTranscriber(mock_text="Patient has headache...")
        result = transcriber.transcribe("any_file.mp3")
        # or async:
        result = await transcriber.atranscribe("any_file.mp3")
    """
    
    def __init__(self, mock_text: str = "Mock transcription text"):
        self.mock_text = mock_text
        self.call_count = 0
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Return mock transcription result (sync)."""
        self.call_count += 1
        return TranscriptionResult(
            text=self.mock_text,
            language="en",
            duration_seconds=60.0,
            confidence=0.95
        )
    
    async def atranscribe(self, audio_path: str) -> TranscriptionResult:
        """Return mock transcription result (async)."""
        self.call_count += 1
        # Simulate some async delay for realistic testing
        await asyncio.sleep(0.01)
        return TranscriptionResult(
            text=self.mock_text,
            language="en",
            duration_seconds=60.0,
            confidence=0.95
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_transcriber(
    settings: Optional[Settings] = None,
    use_mock: bool = False,
    mock_text: str = ""
) -> TranscriberProtocol:
    """
    Factory function to create the appropriate transcriber.
    
    Factory Pattern Benefits:
    1. Encapsulates creation logic
    2. Easy to switch implementations
    3. Cleaner calling code
    4. Centralized configuration
    
    Args:
        settings: Application settings
        use_mock: If True, returns a mock transcriber
        mock_text: Text to return from mock transcriber
        
    Returns:
        A transcriber instance
    """
    if use_mock:
        logger.info("Creating mock transcriber")
        return MockTranscriber(mock_text=mock_text)
    
    logger.info("Creating Whisper transcriber")
    return WhisperTranscriber(settings=settings)
