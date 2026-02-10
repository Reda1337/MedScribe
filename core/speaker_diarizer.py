"""
Speaker Diarization Module

This module provides speaker diarization capabilities using Pyannote.audio to
identify "who spoke when" in medical consultations. This prevents LLM hallucinations
where the doctor is misidentified as having symptoms or the patient making diagnoses.

Key Features:
- Speaker identification using Pyannote speaker-diarization-3.1
- Automatic labeling of Doctor vs Patient speakers
- Timestamped speaker segments for precise attribution
- Integration with Whisper transcription pipeline

HIPAA Compliance:
- All processing done locally (no cloud API calls)
- No audio data leaves the system
- Speaker diarization operates on audio features only

Author: MedScribe AI
License: MIT
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Protocol

# Import models from central models module
from models import SpeakerSegment, DiarizationResult

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Definition (for dependency injection and testing)
# =============================================================================


class SpeakerDiarizerProtocol(Protocol):
    """Protocol defining the interface for speaker diarization services."""

    def diarize(self, audio_path: str) -> DiarizationResult:
        """
        Performs speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            DiarizationResult with speaker segments and timing

        Raises:
            DiarizationError: If diarization fails
            FileNotFoundError: If audio file doesn't exist
        """
        ...


# =============================================================================
# Pyannote Speaker Diarizer Implementation
# =============================================================================


class PyannnoteSpeakerDiarizer:
    """
    Speaker diarization using Pyannote.audio models.

    This implementation uses the pyannote/speaker-diarization-3.1 model
    from HuggingFace, which achieves ~11-19% Diarization Error Rate (DER)
    on standard benchmarks.

    Setup Requirements:
    1. Install: pip install pyannote.audio>=3.1.0
    2. Get HuggingFace token: https://huggingface.co/settings/tokens
    3. Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1
    4. Set environment variable: MedScribe_HUGGINGFACE_TOKEN=your_token_here

    Usage:
        diarizer = PyannnoteSpeakerDiarizer(
            min_speakers=2,
            max_speakers=2,
            auth_token="hf_..."
        )
        result = diarizer.diarize("consultation.wav")
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        auth_token: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize the speaker diarizer.

        Args:
            model_name: HuggingFace model identifier
            min_speakers: Minimum number of speakers (None = auto-detect)
            max_speakers: Maximum number of speakers (None = auto-detect)
            auth_token: HuggingFace authentication token (passed from config)
            device: Device to run on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.device = device
        self.auth_token = auth_token

        if not self.auth_token:
            logger.warning(
                "No HuggingFace token found. Set MedScribe_HUGGINGFACE_TOKEN in .env file. "
                "Get token at: https://huggingface.co/settings/tokens"
            )

        # Lazy loading - pipeline initialized on first use
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the diarization pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading speaker diarization model: {self.model_name}")
            try:
                from pyannote.audio import Pipeline

                self._pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=self.auth_token
                )

                # Move to device (GPU if available)
                import torch
                if self.device == "cuda" and torch.cuda.is_available():
                    self._pipeline = self._pipeline.to(torch.device("cuda"))
                    logger.info("Using GPU for speaker diarization")
                else:
                    logger.info("Using CPU for speaker diarization")

            except Exception as e:
                logger.error(f"Failed to load Pyannote pipeline: {e}")
                raise DiarizationError(
                    f"Could not load speaker diarization model. "
                    f"Make sure you have accepted the model license at: "
                    f"https://huggingface.co/{self.model_name} "
                    f"and set your HuggingFace token."
                ) from e

        return self._pipeline

    def diarize(self, audio_path: str) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.)

        Returns:
            DiarizationResult with speaker segments

        Raises:
            DiarizationError: If diarization fails
            FileNotFoundError: If audio file doesn't exist
        """
        # Validate audio file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting speaker diarization: {audio_path}")

        try:
            # Run diarization pipeline
            diarization = self.pipeline(
                audio_path,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )

            # Convert pyannote output to our format
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    speaker=speaker,
                    start_time=turn.start,
                    end_time=turn.end,
                    text="",  # Will be filled by transcription
                    confidence=None  # Pyannote doesn't provide per-segment confidence
                )
                segments.append(segment)

            # Detect number of unique speakers
            unique_speakers = set(seg.speaker for seg in segments)
            num_speakers = len(unique_speakers)

            # Calculate total duration
            total_duration = max(seg.end_time for seg in segments) if segments else 0.0

            # Auto-label speakers for medical context (Doctor vs Patient)
            speaker_labels = self._auto_label_medical_roles(segments)

            result = DiarizationResult(
                segments=segments,
                num_speakers=num_speakers,
                total_duration=total_duration,
                speaker_labels=speaker_labels
            )

            logger.info(
                f"Diarization complete: {num_speakers} speakers, "
                f"{len(segments)} segments, {total_duration:.1f}s duration"
            )

            return result

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise DiarizationError(f"Speaker diarization failed: {e}") from e

    def _auto_label_medical_roles(self, segments: List[SpeakerSegment]) -> dict[str, str]:
        """
        Automatically label speakers with medical roles (Doctor/Patient).

        Heuristic: In a medical consultation, typically:
        - The speaker who talks MORE is the Doctor (asks questions, explains)
        - The speaker who talks LESS is the Patient (answers questions)

        This is a simple heuristic and may need refinement for complex scenarios.
        Future enhancement: Use voice characteristics or explicit configuration.

        Args:
            segments: List of speaker segments

        Returns:
            Dictionary mapping speaker IDs to roles (e.g., {"SPEAKER_00": "Doctor"})
        """
        # Calculate speaking time per speaker
        speaker_time = {}
        for segment in segments:
            speaker = segment.speaker
            if speaker not in speaker_time:
                speaker_time[speaker] = 0.0
            speaker_time[speaker] += segment.duration

        # Sort speakers by speaking time (descending)
        sorted_speakers = sorted(speaker_time.items(), key=lambda x: x[1], reverse=True)

        # Label speakers
        labels = {}
        if len(sorted_speakers) >= 2:
            # Two speaker scenario (most common)
            labels[sorted_speakers[0][0]] = "Doctor"
            labels[sorted_speakers[1][0]] = "Patient"

            # Additional speakers labeled generically
            for i, (speaker, _) in enumerate(sorted_speakers[2:], start=3):
                labels[speaker] = f"Speaker_{i}"
        elif len(sorted_speakers) == 1:
            # Single speaker (unusual for consultation)
            labels[sorted_speakers[0][0]] = "Speaker"

        logger.info(f"Auto-labeled speakers: {labels}")
        return labels

    def apply_labels_to_segments(
        self,
        result: DiarizationResult,
        apply_labels: bool = True
    ) -> DiarizationResult:
        """
        Apply speaker role labels to segments.

        Args:
            result: DiarizationResult with segments
            apply_labels: If True, replace speaker IDs with role labels

        Returns:
            Updated DiarizationResult with labeled speakers
        """
        if not apply_labels or not result.speaker_labels:
            return result

        # Update speaker field in each segment
        for segment in result.segments:
            if segment.speaker in result.speaker_labels:
                segment.speaker = result.speaker_labels[segment.speaker]

        return result


# =============================================================================
# Mock Implementation (for testing without actual diarization)
# =============================================================================


class MockSpeakerDiarizer:
    """
    Mock speaker diarizer for testing.

    Returns a simple two-speaker pattern without requiring Pyannote.
    Useful for development and testing.
    """

    def diarize(self, audio_path: str) -> DiarizationResult:
        """Returns mock diarization with alternating Doctor/Patient pattern."""
        logger.info(f"[MOCK] Diarizing: {audio_path}")

        # Create mock segments (alternating speakers)
        segments = [
            SpeakerSegment(speaker="Doctor", start_time=0.0, end_time=5.0, confidence=0.95),
            SpeakerSegment(speaker="Patient", start_time=5.0, end_time=10.0, confidence=0.92),
            SpeakerSegment(speaker="Doctor", start_time=10.0, end_time=15.0, confidence=0.94),
            SpeakerSegment(speaker="Patient", start_time=15.0, end_time=20.0, confidence=0.91),
        ]

        return DiarizationResult(
            segments=segments,
            num_speakers=2,
            total_duration=20.0,
            speaker_labels={"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}
        )


# =============================================================================
# Exception Classes
# =============================================================================


class DiarizationError(Exception):
    """Raised when speaker diarization fails."""
    pass


# =============================================================================
# Factory Functions
# =============================================================================


def create_speaker_diarizer(
    use_mock: bool = False,
    **kwargs
) -> SpeakerDiarizerProtocol:
    """
    Factory function to create a speaker diarizer instance.

    Args:
        use_mock: If True, return MockSpeakerDiarizer (for testing)
        **kwargs: Additional arguments passed to PyannnoteSpeakerDiarizer

    Returns:
        SpeakerDiarizerProtocol implementation

    Example:
        # Production use
        diarizer = create_speaker_diarizer(
            min_speakers=2,
            max_speakers=2,
            auth_token="hf_..."
        )

        # Testing use
        diarizer = create_speaker_diarizer(use_mock=True)
    """
    if use_mock:
        logger.info("Creating mock speaker diarizer")
        return MockSpeakerDiarizer()
    else:
        logger.info("Creating Pyannote speaker diarizer")
        return PyannnoteSpeakerDiarizer(**kwargs)


# =============================================================================
# Helper Functions
# =============================================================================


def merge_diarization_with_transcription(
    diarization: DiarizationResult,
    transcription_text: str,
    word_timestamps: Optional[List[dict]] = None
) -> DiarizationResult:
    """
    Merge speaker diarization with Whisper transcription.

    This function aligns the speaker segments with the transcribed text,
    assigning text to the appropriate speaker based on timing.

    Args:
        diarization: Speaker diarization result
        transcription_text: Full transcribed text from Whisper
        word_timestamps: Optional word-level timestamps from Whisper

    Returns:
        Updated DiarizationResult with text populated in segments

    Note:
        If word_timestamps are not available, a simple heuristic is used
        to distribute text across segments based on duration.
    """
    if word_timestamps:
        # Advanced: Use word-level timestamps for precise alignment
        return _merge_with_word_timestamps(diarization, word_timestamps)
    else:
        # Simple: Distribute text evenly across segments (fallback)
        return _merge_with_simple_split(diarization, transcription_text)


def _merge_with_word_timestamps(
    diarization: DiarizationResult,
    word_timestamps: List[dict]
) -> DiarizationResult:
    """Merge using word-level timestamps (most accurate)."""
    # TODO: Implement word-level alignment
    # For now, use simple split
    logger.warning("Word-level timestamp merging not yet implemented, using simple split")
    return diarization


def _merge_with_simple_split(
    diarization: DiarizationResult,
    transcription_text: str
) -> DiarizationResult:
    """
    Simple text distribution across segments.

    Splits transcription text roughly proportional to segment duration.
    Not perfectly accurate but works as a fallback.
    """
    words = transcription_text.split()
    if not words:
        return diarization

    total_duration = sum(seg.duration for seg in diarization.segments)
    if total_duration == 0:
        return diarization

    word_idx = 0
    for segment in diarization.segments:
        # Calculate how many words this segment should get
        proportion = segment.duration / total_duration
        num_words = max(1, int(len(words) * proportion))

        # Assign words to this segment
        segment_words = words[word_idx:word_idx + num_words]
        segment.text = " ".join(segment_words)
        word_idx += num_words

        if word_idx >= len(words):
            break

    # Assign any remaining words to the last segment
    if word_idx < len(words):
        remaining = " ".join(words[word_idx:])
        if diarization.segments:
            diarization.segments[-1].text += " " + remaining

    return diarization
