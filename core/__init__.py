"""
Core Processing Module
======================

Contains the main pipeline and processing components for MedScribe AI:
- pipeline: Main orchestration
- transcriber: Audio transcription with Whisper
- speaker_diarizer: Speaker identification
- soap_generator: SOAP note generation with LLM
- prompts: LLM prompt templates
"""

from core.pipeline import MedicalDocumentationPipeline, create_pipeline, save_result_to_file
from core.soap_generator import OllamaSOAPGenerator, create_soap_generator
from core.transcriber import WhisperTranscriber, create_transcriber
from core.speaker_diarizer import PyannnoteSpeakerDiarizer, create_speaker_diarizer, merge_diarization_with_transcription

__all__ = [
    'MedicalDocumentationPipeline',
    'create_pipeline',
    'save_result_to_file',
    'OllamaSOAPGenerator',
    'create_soap_generator',
    'WhisperTranscriber',
    'create_transcriber',
    'PyannnoteSpeakerDiarizer',
    'create_speaker_diarizer',
    'merge_diarization_with_transcription',
]
