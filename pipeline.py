"""
Processing Pipeline for MedScribe AI
==================================

This module provides the main orchestration layer that combines
transcription and SOAP generation into a single, coherent workflow.

Architecture Pattern: Pipeline
------------------------------
A pipeline is a series of processing stages where:
1. Each stage transforms data
2. Output of one stage is input to the next
3. Stages are independent and reusable

Benefits:
- Clear flow of data
- Easy to add/remove stages
- Simple error handling at each stage
- Easy to test individual stages

Our Pipeline:
Audio File → [Transcriber] → Transcript → [SOAP Generator] → SOAP Note
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Awaitable, Union
import json

from config import Settings, get_settings
from models import (
    ProcessingResult,
    ProcessingStatus,
    TranscriptionResult,
    SOAPNote,
    AudioPath,
)
from transcriber import (
    TranscriberProtocol,
    WhisperTranscriber,
    create_transcriber,
)
from soap_generator import (
    SOAPGeneratorProtocol,
    OllamaSOAPGenerator,
    create_soap_generator,
)
from exceptions import MedScribeError


# Set up module logger
logger = logging.getLogger(__name__)


# Type aliases for progress callbacks
ProgressCallback = Callable[[ProcessingStatus, str, int], None]
AsyncProgressCallback = Callable[[ProcessingStatus, str, int], Awaitable[None]]


class _ProgressHelper:
    """
    Internal helper for calculating granular progress percentages.

    Keeps progress math clean and separate from pipeline orchestration logic.
    Progress is calculated based on stage weights that sum to 100%.
    """

    # Stage weights (must sum to 100)
    TRANSCRIPTION_WEIGHT = 50  # Transcription: 0-50%
    GENERATION_WEIGHT = 50     # SOAP generation: 50-100%

    @staticmethod
    def transcription_progress(stage_percent: int) -> int:
        """
        Calculate overall progress for transcription stage.

        Args:
            stage_percent: Progress within transcription stage (0-100)

        Returns:
            Overall progress percentage (0-50)
        """
        return stage_percent * _ProgressHelper.TRANSCRIPTION_WEIGHT // 100

    @staticmethod
    def generation_progress(stage_percent: int) -> int:
        """
        Calculate overall progress for SOAP generation stage.

        Args:
            stage_percent: Progress within generation stage (0-100)

        Returns:
            Overall progress percentage (50-100)
        """
        base = _ProgressHelper.TRANSCRIPTION_WEIGHT
        offset = stage_percent * _ProgressHelper.GENERATION_WEIGHT // 100
        return base + offset


class MedicalDocumentationPipeline:
    """
    Main pipeline for processing medical audio into SOAP notes.
    
    This class is the primary entry point for using MedScribe AI.
    It orchestrates the transcription and SOAP generation services.
    
    Design Principles:
    -----------------
    1. Dependency Injection: Services injected for testability
    2. Single Responsibility: Only orchestrates, doesn't implement
    3. Open/Closed: Easy to extend with new stages
    4. Error Handling: Comprehensive error capture and reporting
    
    Usage:
        pipeline = MedicalDocumentationPipeline()
        result = pipeline.process("path/to/audio.mp3")
        print(result.soap_note.to_formatted_string())
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        transcriber: Optional[TranscriberProtocol] = None,
        soap_generator: Optional[SOAPGeneratorProtocol] = None,
    ):
        """
        Initialize the pipeline with optional dependencies.
        
        Dependency Injection Pattern:
        - If dependencies are provided, use them (testing/customization)
        - If not, create defaults (production use)
        
        This allows:
        1. Easy unit testing with mocks
        2. Custom configurations
        3. Reuse of existing service instances
        
        Args:
            settings: Application settings
            transcriber: Transcription service
            soap_generator: SOAP note generation service
        """
        self.settings = settings or get_settings()
        
        # Lazy initialization - services created when first needed
        self._transcriber = transcriber
        self._soap_generator = soap_generator
        
        logger.info("MedicalDocumentationPipeline initialized")
    
    @property
    def transcriber(self) -> TranscriberProtocol:
        """Lazy-load the transcriber service."""
        if self._transcriber is None:
            self._transcriber = create_transcriber(settings=self.settings)
        return self._transcriber
    
    @property
    def soap_generator(self) -> SOAPGeneratorProtocol:
        """Lazy-load the SOAP generator service."""
        if self._soap_generator is None:
            self._soap_generator = create_soap_generator(settings=self.settings)
        return self._soap_generator
    
    def process(
        self,
        audio_path: AudioPath,
        progress_callback: Optional[ProgressCallback] = None
    ) -> ProcessingResult:
        """
        Process an audio file and generate a SOAP note.
        
        This is the main entry point for the pipeline. It:
        1. Creates a tracking result object
        2. Runs transcription
        3. Generates SOAP note
        4. Handles errors at each stage
        
        Args:
            audio_path: Path to the audio file
            progress_callback: Optional callback for progress updates
                              Signature: (status: ProcessingStatus, message: str) -> None
        
        Returns:
            ProcessingResult containing transcription and SOAP note
        
        Example:
            def on_progress(status, message):
                print(f"[{status.value}] {message}")
            
            result = pipeline.process("audio.mp3", progress_callback=on_progress)
        """
        # Generate unique ID for this processing job
        job_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        logger.info(f"[{job_id}] Starting pipeline for: {audio_path}")
        
        # Create result object to track progress
        result = ProcessingResult(
            id=job_id,
            audio_file_path=audio_path,
            status=ProcessingStatus.PENDING
        )
        
        try:
            # Stage 1: Transcription
            result = self._run_transcription(result, progress_callback)
            
            # Stage 2: SOAP Generation
            result = self._run_soap_generation(result, progress_callback)
            
            # Mark as completed
            result.status = ProcessingStatus.COMPLETED
            result.completed_at = datetime.now()
            result.processing_time_seconds = (
                result.completed_at - start_time
            ).total_seconds()
            
            self._notify_progress(
                progress_callback,
                ProcessingStatus.COMPLETED,
                f"Processing complete in {result.processing_time_seconds:.1f}s",
                100
            )
            
            logger.info(
                f"[{job_id}] Pipeline completed successfully in "
                f"{result.processing_time_seconds:.1f}s"
            )
            
        except MedScribeError as e:
            # Known errors - we have structured information
            result.status = ProcessingStatus.FAILED
            result.error_message = e.message
            result.completed_at = datetime.now()
            
            self._notify_progress(
                progress_callback,
                ProcessingStatus.FAILED,
                f"Error: {e.message}",
                0
            )

            logger.error(f"[{job_id}] Pipeline failed: {e.message}")

        except Exception as e:
            # Unexpected errors
            result.status = ProcessingStatus.FAILED
            result.error_message = f"Unexpected error: {str(e)}"
            result.completed_at = datetime.now()

            self._notify_progress(
                progress_callback,
                ProcessingStatus.FAILED,
                f"Unexpected error: {str(e)}",
                0
            )
            
            logger.exception(f"[{job_id}] Unexpected error in pipeline")
        
        return result
    
    def _run_transcription(
        self,
        result: ProcessingResult,
        progress_callback: Optional[ProgressCallback]
    ) -> ProcessingResult:
        """
        Run the transcription stage.

        This is a separate method for:
        1. Cleaner code organization
        2. Easier testing of individual stages
        3. Potential for parallel processing (future)
        """
        # Starting transcription - 0% of stage
        self._notify_progress(
            progress_callback,
            ProcessingStatus.TRANSCRIBING,
            "Starting transcription...",
            _ProgressHelper.transcription_progress(0)
        )

        result.status = ProcessingStatus.TRANSCRIBING

        logger.info(f"[{result.id}] Starting transcription")

        # Mid-transcription progress - 50% of stage
        self._notify_progress(
            progress_callback,
            ProcessingStatus.TRANSCRIBING,
            "Processing audio with Whisper...",
            _ProgressHelper.transcription_progress(50)
        )

        transcription = self.transcriber.transcribe(result.audio_file_path)
        result.transcription = transcription

        # Transcription complete - 100% of stage
        self._notify_progress(
            progress_callback,
            ProcessingStatus.TRANSCRIBING,
            "Transcription complete",
            _ProgressHelper.transcription_progress(100)
        )

        logger.info(
            f"[{result.id}] Transcription complete: "
            f"{len(transcription.text)} chars, {transcription.duration_seconds:.1f}s"
        )

        return result
    
    def _run_soap_generation(
        self,
        result: ProcessingResult,
        progress_callback: Optional[ProgressCallback]
    ) -> ProcessingResult:
        """
        Run the SOAP generation stage.
        
        Requires transcription to be complete first.
        
        Multi-language support: The detected language from transcription
        is passed to the SOAP generator to ensure the note is generated
        in the same language as the original audio.
        """
        if not result.transcription:
            raise ValueError("Cannot generate SOAP without transcription")

        # Starting SOAP generation - 0% of stage
        self._notify_progress(
            progress_callback,
            ProcessingStatus.GENERATING,
            "Starting SOAP generation...",
            _ProgressHelper.generation_progress(0)
        )

        result.status = ProcessingStatus.GENERATING

        # Extract detected language, default to English if not available
        detected_language = result.transcription.language or "en"

        logger.info(
            f"[{result.id}] Starting SOAP generation "
            f"(detected language: {detected_language})"
        )

        # Mid-generation progress - 30% of stage
        self._notify_progress(
            progress_callback,
            ProcessingStatus.GENERATING,
            "Analyzing transcription...",
            _ProgressHelper.generation_progress(30)
        )

        # Before LLM call - 60% of stage
        self._notify_progress(
            progress_callback,
            ProcessingStatus.GENERATING,
            "Generating SOAP note with LLM...",
            _ProgressHelper.generation_progress(60)
        )

        # Pass detected language to generator for multi-language support
        soap_note = self.soap_generator.generate(
            result.transcription.text,
            language=detected_language
        )
        result.soap_note = soap_note

        # SOAP generation complete - 100% of stage
        self._notify_progress(
            progress_callback,
            ProcessingStatus.GENERATING,
            "SOAP note generated",
            _ProgressHelper.generation_progress(100)
        )
        
        logger.info(f"[{result.id}] SOAP note generated in {detected_language}")
        
        return result
    
    def _notify_progress(
        self,
        callback: Optional[ProgressCallback],
        status: ProcessingStatus,
        message: str,
        progress: int = 0
    ) -> None:
        """
        Notify progress callback if provided.

        This is a helper to safely call the callback without
        crashing if it fails. Exception isolation ensures
        callback errors don't break the pipeline.

        Args:
            callback: Progress callback function
            status: Current processing status
            message: Human-readable progress message
            progress: Progress percentage (0-100), defaults to 0
        """
        if callback:
            try:
                callback(status, message, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def transcribe_only(self, audio_path: AudioPath) -> TranscriptionResult:
        """
        Transcribe audio without generating SOAP note.
        
        Useful for:
        - Debugging transcription issues
        - Getting transcription to review before SOAP
        - Processing where SOAP is not needed
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            TranscriptionResult with transcribed text
        """
        logger.info(f"Transcribe-only mode for: {audio_path}")
        return self.transcriber.transcribe(audio_path)
    
    def generate_soap_only(self, transcription: str) -> SOAPNote:
        """
        Generate SOAP note from existing transcription.
        
        Useful for:
        - Re-processing existing transcriptions
        - Testing different prompts
        - Manual transcription input
        
        Args:
            transcription: The medical consultation transcript
            
        Returns:
            SOAPNote object
        """
        logger.info("SOAP-only mode for provided transcription")
        return self.soap_generator.generate(transcription)

    # =========================================================================
    # Async Methods (for FastAPI/Web Frameworks)
    # =========================================================================

    async def aprocess(
        self,
        audio_path: AudioPath,
        progress_callback: Optional[Union[ProgressCallback, AsyncProgressCallback]] = None
    ) -> ProcessingResult:
        """
        Async version of process() for use with FastAPI/async frameworks.

        This is the primary entry point for API/web applications. It:
        1. Creates a tracking result object
        2. Runs transcription asynchronously (thread pool for CPU-bound work)
        3. Generates SOAP note asynchronously (native async via LangChain)
        4. Handles errors at each stage

        Benefits of async:
        - Non-blocking: Other requests can be served while processing
        - Efficient: Event loop handles I/O concurrency
        - Scalable: Single process can handle many concurrent requests

        Args:
            audio_path: Path to the audio file
            progress_callback: Optional callback for progress updates.
                              Can be sync or async function.

        Returns:
            ProcessingResult containing transcription and SOAP note
        """
        job_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        logger.info(f"[{job_id}] Starting async pipeline for: {audio_path}")

        result = ProcessingResult(
            id=job_id,
            audio_file_path=audio_path,
            status=ProcessingStatus.PENDING
        )

        try:
            # Stage 1: Async Transcription
            result = await self._arun_transcription(result, progress_callback)

            # Stage 2: Async SOAP Generation
            result = await self._arun_soap_generation(result, progress_callback)

            # Mark as completed
            result.status = ProcessingStatus.COMPLETED
            result.completed_at = datetime.now()
            result.processing_time_seconds = (
                result.completed_at - start_time
            ).total_seconds()

            await self._anotify_progress(
                progress_callback,
                ProcessingStatus.COMPLETED,
                f"Processing complete in {result.processing_time_seconds:.1f}s",
                100
            )

            logger.info(
                f"[{job_id}] Async pipeline completed successfully in "
                f"{result.processing_time_seconds:.1f}s"
            )

        except MedScribeError as e:
            result.status = ProcessingStatus.FAILED
            result.error_message = e.message
            result.completed_at = datetime.now()

            await self._anotify_progress(
                progress_callback,
                ProcessingStatus.FAILED,
                f"Error: {e.message}",
                0
            )

            logger.error(f"[{job_id}] Async pipeline failed: {e.message}")

        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.error_message = f"Unexpected error: {str(e)}"
            result.completed_at = datetime.now()

            await self._anotify_progress(
                progress_callback,
                ProcessingStatus.FAILED,
                f"Unexpected error: {str(e)}",
                0
            )

            logger.exception(f"[{job_id}] Unexpected error in async pipeline")

        return result

    async def _arun_transcription(
        self,
        result: ProcessingResult,
        progress_callback: Optional[Union[ProgressCallback, AsyncProgressCallback]]
    ) -> ProcessingResult:
        """
        Run the transcription stage asynchronously.

        Uses asyncio.to_thread internally to run the blocking
        Whisper model without freezing the event loop.
        """
        # Starting transcription - 0% of stage
        await self._anotify_progress(
            progress_callback,
            ProcessingStatus.TRANSCRIBING,
            "Starting transcription...",
            _ProgressHelper.transcription_progress(0)
        )

        result.status = ProcessingStatus.TRANSCRIBING

        logger.info(f"[{result.id}] Starting async transcription")

        # Mid-transcription progress - 50% of stage
        await self._anotify_progress(
            progress_callback,
            ProcessingStatus.TRANSCRIBING,
            "Processing audio with Whisper...",
            _ProgressHelper.transcription_progress(50)
        )

        # Use the async transcribe method
        transcription = await self.transcriber.atranscribe(result.audio_file_path)
        result.transcription = transcription

        # Transcription complete - 100% of stage
        await self._anotify_progress(
            progress_callback,
            ProcessingStatus.TRANSCRIBING,
            "Transcription complete",
            _ProgressHelper.transcription_progress(100)
        )

        logger.info(
            f"[{result.id}] Async transcription complete: "
            f"{len(transcription.text)} chars, {transcription.duration_seconds:.1f}s"
        )

        return result

    async def _arun_soap_generation(
        self,
        result: ProcessingResult,
        progress_callback: Optional[Union[ProgressCallback, AsyncProgressCallback]]
    ) -> ProcessingResult:
        """
        Run the SOAP generation stage asynchronously.

        Uses LangChain's native async support (ainvoke) for
        non-blocking LLM calls.
        """
        if not result.transcription:
            raise ValueError("Cannot generate SOAP without transcription")

        # Starting SOAP generation - 0% of stage
        await self._anotify_progress(
            progress_callback,
            ProcessingStatus.GENERATING,
            "Starting SOAP generation...",
            _ProgressHelper.generation_progress(0)
        )

        result.status = ProcessingStatus.GENERATING

        detected_language = result.transcription.language or "en"

        logger.info(
            f"[{result.id}] Starting async SOAP generation "
            f"(detected language: {detected_language})"
        )

        # Mid-generation progress - 30% of stage
        await self._anotify_progress(
            progress_callback,
            ProcessingStatus.GENERATING,
            "Analyzing transcription...",
            _ProgressHelper.generation_progress(30)
        )

        # Before LLM call - 60% of stage
        await self._anotify_progress(
            progress_callback,
            ProcessingStatus.GENERATING,
            "Generating SOAP note with LLM...",
            _ProgressHelper.generation_progress(60)
        )

        # Use the async generate method
        soap_note = await self.soap_generator.agenerate(
            result.transcription.text,
            language=detected_language
        )
        result.soap_note = soap_note

        # SOAP generation complete - 100% of stage
        await self._anotify_progress(
            progress_callback,
            ProcessingStatus.GENERATING,
            "SOAP note generated",
            _ProgressHelper.generation_progress(100)
        )

        logger.info(f"[{result.id}] Async SOAP note generated in {detected_language}")

        return result

    async def _anotify_progress(
        self,
        callback: Optional[Union[ProgressCallback, AsyncProgressCallback]],
        status: ProcessingStatus,
        message: str,
        progress: int = 0
    ) -> None:
        """
        Notify progress callback if provided (supports sync and async callbacks).

        Args:
            callback: Progress callback function (sync or async)
            status: Current processing status
            message: Human-readable progress message
            progress: Progress percentage (0-100), defaults to 0
        """
        if callback:
            try:
                # Check if callback is a coroutine function
                if asyncio.iscoroutinefunction(callback):
                    await callback(status, message, progress)
                else:
                    callback(status, message, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def atranscribe_only(self, audio_path: AudioPath) -> TranscriptionResult:
        """
        Async version of transcribe_only().

        Transcribe audio without generating SOAP note.
        """
        logger.info(f"Async transcribe-only mode for: {audio_path}")
        return await self.transcriber.atranscribe(audio_path)

    async def agenerate_soap_only(self, transcription: str, language: str = "en") -> SOAPNote:
        """
        Async version of generate_soap_only().

        Generate SOAP note from existing transcription.
        """
        logger.info("Async SOAP-only mode for provided transcription")
        return await self.soap_generator.agenerate(transcription, language)


def save_result_to_file(
    result: ProcessingResult,
    output_dir: str = "./output"
) -> dict[str, str]:
    """
    Save processing result to files.
    
    This is a utility function (not part of the class) for saving results.
    Keeping it separate follows the Single Responsibility Principle.
    
    Saves:
    1. Full result as JSON (for API/database)
    2. SOAP note as formatted text (for reading)
    3. Transcription as plain text (for reference)
    
    Args:
        result: The ProcessingResult to save
        output_dir: Directory to save files in
        
    Returns:
        Dict of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = f"MedScribe_{result.id}"
    saved_files = {}
    
    # Save full result as JSON
    json_path = output_path / f"{base_name}_result.json"
    with open(json_path, 'w') as f:
        json.dump(result.model_dump(mode='json'), f, indent=2, default=str)
    saved_files['json'] = str(json_path)
    
    # Save SOAP note as formatted text
    if result.soap_note:
        soap_path = output_path / f"{base_name}_soap.txt"
        with open(soap_path, 'w') as f:
            f.write(result.soap_note.to_formatted_string())
        saved_files['soap'] = str(soap_path)
    
    # Save transcription
    if result.transcription:
        trans_path = output_path / f"{base_name}_transcription.txt"
        with open(trans_path, 'w') as f:
            f.write(result.transcription.text)
        saved_files['transcription'] = str(trans_path)
    
    logger.info(f"Saved results to {output_dir}: {list(saved_files.keys())}")
    
    return saved_files


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_process(audio_path: str) -> SOAPNote:
    """
    Quick one-liner to process audio file.
    
    This is a convenience function for simple use cases.
    For production code, use the Pipeline class directly.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        SOAPNote object
        
    Example:
        soap = quick_process("consultation.mp3")
        print(soap.to_formatted_string())
    """
    pipeline = MedicalDocumentationPipeline()
    result = pipeline.process(audio_path)
    
    if result.status == ProcessingStatus.FAILED:
        raise MedScribeError(result.error_message or "Processing failed")
    
    return result.soap_note


def create_pipeline(
    settings: Optional[Settings] = None
) -> MedicalDocumentationPipeline:
    """
    Factory function to create a configured pipeline instance.
    
    This is the recommended way to create a pipeline for API usage,
    as it ensures proper initialization with settings.
    
    Args:
        settings: Optional custom settings. Uses default if not provided.
        
    Returns:
        MedicalDocumentationPipeline: Configured pipeline instance
        
    Example:
        pipeline = create_pipeline()
        result = await pipeline.aprocess("audio.mp3")
    """
    if settings is None:
        settings = get_settings()
    
    return MedicalDocumentationPipeline(settings=settings)
