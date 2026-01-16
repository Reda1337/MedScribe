"""
Domain Models for MedScribe AI
============================

This module defines the core data structures used throughout the application.
We use Pydantic for several important reasons:

1. **Validation**: Automatically validates data types and constraints
2. **Serialization**: Easy conversion to/from JSON (crucial for future API)
3. **Documentation**: Self-documenting with type hints
4. **Immutability**: Can enforce immutable data structures

Design Principle: These models are "pure" - they have no dependencies on
external services, databases, or frameworks. This makes them highly reusable
and testable.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """
    Enum for tracking the status of audio processing.
    
    Using str, Enum allows JSON serialization while maintaining type safety.
    This is important for:
    - API responses (future)
    - Logging and monitoring
    - State management
    """
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class SpeakerSegment(BaseModel):
    """
    Represents a single speaker segment with timing information.

    Used for speaker diarization to identify "who spoke when" in medical consultations.
    This prevents LLM hallucinations where the doctor is misidentified as having symptoms.

    Attributes:
        speaker: Speaker identifier (e.g., "Doctor", "Patient", "SPEAKER_00")
        start_time: Segment start time in seconds
        end_time: Segment end time in seconds
        text: Transcribed text for this segment
        confidence: Diarization confidence score (0.0-1.0)
    """
    speaker: str = Field(..., description="Speaker identifier or role")
    start_time: float = Field(..., description="Segment start time in seconds")
    end_time: float = Field(..., description="Segment end time in seconds")
    text: str = Field(default="", description="Transcribed text for this segment")
    confidence: Optional[float] = Field(
        default=None,
        description="Diarization confidence score (0.0-1.0)"
    )

    @property
    def duration(self) -> float:
        """Returns the duration of this segment in seconds."""
        return self.end_time - self.start_time

    def to_labeled_text(self) -> str:
        """
        Returns formatted text with speaker label.

        Example:
            "Doctor: Patient reports chest pain for 2 days."
        """
        return f"{self.speaker}: {self.text}" if self.text else ""

    class Config:
        from_attributes = True


class DiarizationResult(BaseModel):
    """
    Complete speaker diarization result for an audio file.

    Attributes:
        segments: List of speaker segments with timing
        num_speakers: Number of unique speakers detected
        total_duration: Total audio duration in seconds
        speaker_labels: Mapping of speaker IDs to roles (e.g., SPEAKER_00 -> Doctor)
    """
    segments: List[SpeakerSegment] = Field(
        default_factory=list,
        description="List of speaker segments with timing"
    )
    num_speakers: int = Field(..., description="Number of unique speakers detected")
    total_duration: float = Field(..., description="Total audio duration in seconds")
    speaker_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of speaker IDs to roles"
    )

    def get_formatted_transcript(self) -> str:
        """
        Returns a formatted transcript with speaker labels.

        Example output:
            Doctor: Good morning, what brings you in today?
            Patient: I've been having chest pain for the past two days.
            Doctor: Can you describe the pain for me?
        """
        lines = []
        for segment in self.segments:
            if segment.text.strip():
                lines.append(segment.to_labeled_text())
        return "\n".join(lines)

    def get_speaker_statistics(self) -> dict[str, float]:
        """Returns speaking time per speaker in seconds."""
        stats = {}
        for segment in self.segments:
            speaker = segment.speaker
            if speaker not in stats:
                stats[speaker] = 0.0
            stats[speaker] += segment.duration
        return stats

    class Config:
        from_attributes = True


class TranscriptionResult(BaseModel):
    """
    Represents the output of the transcription service.

    Enhanced with speaker diarization support to identify "who spoke when"
    in medical consultations.

    Keeping this separate from the SOAP note allows us to:
    1. Cache transcriptions independently
    2. Re-process transcriptions with different prompts
    3. Debug issues at each stage separately
    """
    text: str = Field(..., description="The transcribed text from audio")
    language: str = Field(default="en", description="Detected language code")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    confidence: Optional[float] = Field(
        default=None,
        description="Transcription confidence score (0-1)"
    )

    # Phase 1: Speaker diarization support
    speaker_segments: Optional[List[SpeakerSegment]] = Field(
        default=None,
        description="Speaker-labeled segments (populated if diarization enabled)"
    )
    diarization: Optional[DiarizationResult] = Field(
        default=None,
        description="Full diarization result with speaker statistics"
    )

    def get_formatted_transcript(self) -> str:
        """
        Returns formatted transcript with speaker labels if available.

        Falls back to plain text if diarization not performed.
        """
        if self.diarization:
            return self.diarization.get_formatted_transcript()
        elif self.speaker_segments:
            lines = [seg.to_labeled_text() for seg in self.speaker_segments if seg.text.strip()]
            return "\n".join(lines)
        else:
            return self.text

    class Config:
        # Allows creating from ORM objects (useful for future database integration)
        from_attributes = True


class ClinicalCode(BaseModel):
    """
    Represents a medical billing/diagnosis code.

    Supports ICD-10 (diagnoses) and CPT (procedures) coding standards.

    Attributes:
        code: The actual code (e.g., "I10", "99213")
        code_type: Type of code ("ICD-10", "CPT", "SNOMED")
        description: Human-readable description of the code
        confidence: Confidence score for AI-suggested codes (0.0-1.0)
    """
    code: str = Field(..., description="The medical code")
    code_type: str = Field(..., description="Type of code (ICD-10, CPT, SNOMED)")
    description: str = Field(..., description="Human-readable description")
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for AI-suggested codes (0.0-1.0)"
    )

    class Config:
        from_attributes = True


class ProcessingMetrics(BaseModel):
    """
    Metrics and quality scores for the processing pipeline.

    Used for monitoring, quality assurance, and system improvement.

    Attributes:
        transcription_duration: Time taken for transcription (seconds)
        diarization_duration: Time taken for speaker diarization (seconds)
        soap_generation_duration: Time taken for SOAP note generation (seconds)
        total_tokens_used: Total tokens consumed by LLM
        diarization_confidence: Average diarization confidence score
        soap_completeness_score: Quality score for SOAP note completeness
        medical_terminology_percentage: Percentage of medical terms used
    """
    transcription_duration: Optional[float] = Field(
        default=None,
        description="Time taken for transcription (seconds)"
    )
    diarization_duration: Optional[float] = Field(
        default=None,
        description="Time taken for speaker diarization (seconds)"
    )
    soap_generation_duration: Optional[float] = Field(
        default=None,
        description="Time taken for SOAP note generation (seconds)"
    )
    total_tokens_used: Optional[int] = Field(
        default=None,
        description="Total tokens consumed by LLM"
    )
    diarization_confidence: Optional[float] = Field(
        default=None,
        description="Average diarization confidence score (0.0-1.0)"
    )
    soap_completeness_score: Optional[float] = Field(
        default=None,
        description="Quality score for SOAP note completeness (0.0-1.0)"
    )
    medical_terminology_percentage: Optional[float] = Field(
        default=None,
        description="Percentage of medical terminology used (0.0-100.0)"
    )

    class Config:
        from_attributes = True


class SOAPNote(BaseModel):
    """
    SOAP Note - The standard medical documentation format.

    SOAP stands for:
    - Subjective: Patient's reported symptoms and history
    - Objective: Observable/measurable findings
    - Assessment: Diagnosis or differential diagnoses
    - Plan: Treatment plan and next steps

    This format is used universally in healthcare because it:
    1. Provides consistent structure for documentation
    2. Separates facts from interpretation
    3. Creates clear action items
    4. Supports billing and legal requirements

    Enhanced with Phase 2 features:
    - Clinical coding support (ICD-10, CPT)
    - Structured subsections (HPI, ROS)
    """
    subjective: str = Field(
        ...,
        description="Patient's reported symptoms, history, and concerns"
    )
    objective: str = Field(
        ...,
        description="Observable findings, vital signs, examination results"
    )
    assessment: str = Field(
        ...,
        description="Clinical assessment, diagnosis, or differential diagnoses"
    )
    plan: str = Field(
        ...,
        description="Treatment plan, medications, follow-up instructions"
    )

    # Phase 2: Clinical coding support
    diagnosis_codes: Optional[List[ClinicalCode]] = Field(
        default=None,
        description="ICD-10 diagnosis codes extracted from assessment"
    )
    procedure_codes: Optional[List[ClinicalCode]] = Field(
        default=None,
        description="CPT procedure codes from plan"
    )

    # Phase 7: Quality metrics
    metrics: Optional[ProcessingMetrics] = Field(
        default=None,
        description="Quality and performance metrics"
    )
    
    def to_formatted_string(self) -> str:
        """
        Returns a nicely formatted SOAP note for display or export.
        
        This method encapsulates the presentation logic within the model,
        following the principle of "Tell, Don't Ask" - the object knows
        how to present itself.
        """
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                         SOAP NOTE                                 ║
╠══════════════════════════════════════════════════════════════════╣
║ SUBJECTIVE                                                        ║
╟──────────────────────────────────────────────────────────────────╢
{self._wrap_text(self.subjective)}
╟──────────────────────────────────────────────────────────────────╢
║ OBJECTIVE                                                         ║
╟──────────────────────────────────────────────────────────────────╢
{self._wrap_text(self.objective)}
╟──────────────────────────────────────────────────────────────────╢
║ ASSESSMENT                                                        ║
╟──────────────────────────────────────────────────────────────────╢
{self._wrap_text(self.assessment)}
╟──────────────────────────────────────────────────────────────────╢
║ PLAN                                                              ║
╟──────────────────────────────────────────────────────────────────╢
{self._wrap_text(self.plan)}
╚══════════════════════════════════════════════════════════════════╝
"""
    
    def _wrap_text(self, text: str, width: int = 66) -> str:
        """Helper to wrap text for formatted output."""
        lines = []
        for paragraph in text.split('\n'):
            words = paragraph.split()
            current_line = "║ "
            for word in words:
                if len(current_line) + len(word) + 1 <= width:
                    current_line += word + " "
                else:
                    lines.append(current_line.ljust(67) + "║")
                    current_line = "║ " + word + " "
            if current_line.strip("║ "):
                lines.append(current_line.ljust(67) + "║")
        return '\n'.join(lines) if lines else "║" + " " * 66 + "║"


class ProcessingResult(BaseModel):
    """
    Complete result of processing an audio file.
    
    This is our main "aggregate" - it combines all related data into
    a single coherent unit. This pattern is from Domain-Driven Design (DDD).
    
    Benefits:
    1. Single object to pass around (reduces parameter counts)
    2. Maintains consistency between related data
    3. Easy to serialize for API responses or storage
    """
    id: str = Field(..., description="Unique identifier for this processing job")
    status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current processing status"
    )
    audio_file_path: str = Field(..., description="Path to the original audio file")
    transcription: Optional[TranscriptionResult] = Field(
        default=None,
        description="Transcription result (populated after transcription)"
    )
    soap_note: Optional[SOAPNote] = Field(
        default=None,
        description="Generated SOAP note (populated after generation)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this processing job was created"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When processing completed (success or failure)"
    )
    processing_time_seconds: Optional[float] = Field(
        default=None,
        description="Total time taken to process"
    )
    
    class Config:
        from_attributes = True
        # Allow mutation - we update this object as processing progresses
        frozen = False


# Type alias for cleaner function signatures
AudioPath = str
