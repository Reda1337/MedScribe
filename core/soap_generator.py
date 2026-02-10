"""
SOAP Note Generator for MedScribe AI
==================================

This module converts medical transcriptions into structured SOAP notes
using Ollama (local LLM) with LangChain for orchestration.

Architecture Pattern: Service with Strategy
-------------------------------------------
The generator is designed to:
1. Accept different LLM backends (Ollama, OpenAI, etc.)
2. Support different prompting strategies
3. Handle parsing of LLM output into structured data

Why Ollama + LangChain?
-----------------------
Ollama:
- Free and open source
- Runs locally (data privacy)
- No API costs
- Good model selection (Llama, Mistral, etc.)

LangChain:
- Abstraction over different LLM providers
- Easy to switch models
- Built-in prompt templates
- Retry and error handling
- Future: chains, agents, RAG integration
"""

import asyncio
import logging
import re
from typing import Optional, Protocol

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import Settings, get_settings
from models import SOAPNote, TranscriptionResult
from core.prompts import get_professional_soap_prompt, get_system_prompt
from exceptions import (
    OllamaConnectionError,
    ModelNotFoundError,
    SOAPGenerationError,
)


# Set up module logger
logger = logging.getLogger(__name__)


class SOAPGeneratorProtocol(Protocol):
    """
    Protocol for SOAP note generators.
    
    This allows us to swap implementations:
    - OllamaGenerator: Local LLM
    - OpenAIGenerator: Cloud API (future)
    - MockGenerator: Testing
    """
    
    def generate(self, transcription: str, language: str = "en") -> SOAPNote:
        """
        Generate a SOAP note from transcription text (synchronous).
        
        Args:
            transcription: The medical consultation transcript
            language: ISO 639-1 language code for the output SOAP note.
                     The note will be generated in this language to match
                     the original audio/transcript language.
                     Defaults to "en" (English) for backward compatibility.
            
        Returns:
            Structured SOAPNote object
        """
        ...
    
    async def agenerate(self, transcription: str, language: str = "en") -> SOAPNote:
        """
        Generate a SOAP note from transcription text (asynchronous).
        
        This is the async version for use with FastAPI, aiohttp, etc.
        
        Args:
            transcription: The medical consultation transcript
            language: ISO 639-1 language code for the output SOAP note.
            
        Returns:
            Structured SOAPNote object
        """
        ...


class OllamaSOAPGenerator:
    """
    SOAP note generator using Ollama LLM.

    Enhanced with production-grade prompting (Phases 2-3):
    - Professional clinical documentation standards (HPI/ROS, ICD-10/CPT)
    - Few-shot prompting with 5 curated medical examples
    - Chain-of-Thought reasoning for clinical accuracy
    - Speaker diarization support (Doctor vs Patient attribution)

    This is our primary implementation. It uses LangChain to interact
    with a locally-running Ollama instance.

    Key Design Decisions:
    ---------------------
    1. Lazy initialization: LLM connection only when needed
    2. Structured output parsing: Convert raw text to SOAPNote
    3. Error handling: Graceful handling of LLM issues
    4. Logging: Detailed logging for debugging
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[OllamaLLM] = None
    ):
        """
        Initialize the SOAP generator.
        
        Args:
            settings: Application settings (uses defaults if not provided)
            llm: Pre-configured LLM instance (creates one if not provided)
        
        Dependency Injection allows:
        - Testing with mock LLM
        - Reusing LLM instances
        - Custom configuration
        """
        self.settings = settings or get_settings()
        self._llm = llm
        self._llm_initialized = llm is not None
        
        logger.info(
            f"OllamaSOAPGenerator initialized with model: {self.settings.ollama_model}"
        )
    
    @property
    def llm(self) -> OllamaLLM:
        """
        Lazy-load the LLM instance.
        
        Benefits of lazy loading:
        1. Faster application startup
        2. Fail only when actually needed
        3. Resource efficiency
        """
        if not self._llm_initialized:
            self._initialize_llm()
        return self._llm
    
    def _initialize_llm(self) -> None:
        """
        Initialize the Ollama LLM connection.
        
        This creates the LangChain Ollama wrapper with our configuration.
        """
        try:
            logger.info(
                f"Initializing Ollama LLM: {self.settings.ollama_model} "
                f"at {self.settings.ollama_base_url}"
            )
            
            self._llm = OllamaLLM(
                model=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                temperature=self.settings.ollama_temperature,
                # Context window from settings (supports long transcripts + few-shot examples)
                num_ctx=self.settings.ollama_context_window,
            )
            
            # Test the connection with a simple prompt
            self._test_connection()
            
            self._llm_initialized = True
            logger.info("Ollama LLM initialized successfully")
            
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                raise OllamaConnectionError(
                    url=self.settings.ollama_base_url,
                    original_error=error_msg
                )
            raise
    
    def _test_connection(self) -> None:
        """
        Test the Ollama connection with a simple prompt.
        
        This validates:
        1. Ollama is running
        2. The model is available
        3. Basic inference works
        """
        try:
            # Simple test prompt
            test_response = self._llm.invoke("Say 'OK' if you're working.")
            logger.debug(f"Ollama test response: {test_response[:50]}...")
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "pull" in error_msg:
                raise ModelNotFoundError(self.settings.ollama_model)
            raise
    
    def generate(self, transcription: str, language: str = "en") -> SOAPNote:
        """
        Generate a professional SOAP note from a medical transcription.

        Enhanced with production-grade prompting (Phases 2-3):
        - Uses professional medical documentation standards
        - Includes 5 few-shot examples for format learning
        - Implements Chain-of-Thought reasoning
        - Properly handles speaker-labeled transcripts (Doctor vs Patient)
        - Multi-language support: generates SOAP note in the detected language

        This is the main public method. The process:
        1. Create professional prompt with few-shot examples + CoT + language instruction
        2. Send to LLM with proper system/user message structure
        3. Parse the response into structured SOAPNote

        Args:
            transcription: The medical consultation transcript (may include speaker labels)
            language: ISO 639-1 language code (e.g., "es", "fr", "ru").
                     The SOAP note will be generated in this language to match
                     the original audio. Defaults to "en" for backward compatibility.

        Returns:
            Structured SOAPNote object with professional clinical documentation

        Raises:
            SOAPGenerationError: If generation or parsing fails
        """
        if not transcription or not transcription.strip():
            raise SOAPGenerationError(
                reason="Empty transcription provided",
                transcription_preview=""
            )

        logger.info(
            f"Generating professional SOAP note for transcription "
            f"({len(transcription)} chars, language: {language})"
        )

        try:
            # Step 1: Build the professional prompt with language support
            # get_professional_soap_prompt returns (system_prompt, user_prompt)
            # where user_prompt includes:
            #   - 5 few-shot examples showing professional format
            #   - Chain-of-Thought instructions
            #   - The actual transcription to process
            #   - Language instruction (if not English)
            system_prompt, user_prompt = get_professional_soap_prompt(
                transcription,
                target_language=language
            )

            # Using LangChain's ChatPromptTemplate for structured prompting
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])

            # Step 2: Create the chain
            # Chain pattern: prompt -> llm -> output_parser
            # The professional prompt is ~15K tokens (few-shot examples)
            # but dramatically improves output quality
            chain = prompt | self.llm | StrOutputParser()

            # Step 3: Execute the chain
            logger.debug("Sending request to Ollama with professional prompt (includes few-shot examples)...")
            raw_response = chain.invoke({})  # No variables needed - already in user_prompt

            logger.debug(f"Received response ({len(raw_response)} chars)")

            # Step 4: Parse the response into structured SOAPNote
            soap_note = self._parse_soap_response(raw_response)

            # Step 5: Validate clinical safety and logic
            logger.debug("Validating SOAP note for safety and clinical logic...")

            safety_warnings = self._validate_clinical_safety(soap_note)
            logic_warnings = self._validate_clinical_logic(soap_note)

            all_warnings = safety_warnings + logic_warnings

            # If there are warnings, append them to the PLAN section
            if all_warnings:
                logger.warning(f"SOAP note validation found {len(all_warnings)} issue(s)")

                warning_text = "\n\n" + "="*70 + "\n"
                warning_text += "VALIDATION WARNINGS\n"
                warning_text += "="*70 + "\n"
                warning_text += "\n".join(all_warnings)
                warning_text += "\n" + "="*70

                # Append warnings to plan section
                soap_note.plan = soap_note.plan + warning_text

                # Log each warning
                for warning in all_warnings:
                    logger.warning(warning)
            else:
                logger.info("SOAP note passed all validation checks")

            logger.info("Professional SOAP note generated successfully")
            return soap_note

        except SOAPGenerationError:
            raise
        except Exception as e:
            logger.error(f"SOAP generation failed: {e}")
            raise SOAPGenerationError(
                reason=str(e),
                transcription_preview=transcription[:500]  # Only preview first 500 chars
            )

    async def agenerate(self, transcription: str, language: str = "en") -> SOAPNote:
        """
        Async version of generate() for use with FastAPI/async frameworks.

        This method uses LangChain's native async support (ainvoke) to avoid
        blocking the event loop. This is critical for web servers handling
        multiple concurrent requests.

        Args:
            transcription: The medical consultation transcript
            language: ISO 639-1 language code for the SOAP note output

        Returns:
            Structured SOAPNote object with professional clinical documentation

        Raises:
            SOAPGenerationError: If generation or parsing fails
        """
        if not transcription or not transcription.strip():
            raise SOAPGenerationError(
                reason="Empty transcription provided",
                transcription_preview=""
            )

        logger.info(
            f"Generating professional SOAP note (async) for transcription "
            f"({len(transcription)} chars, language: {language})"
        )

        try:
            # Step 1: Build the professional prompt with language support
            system_prompt, user_prompt = get_professional_soap_prompt(
                transcription,
                target_language=language
            )

            # Using LangChain's ChatPromptTemplate for structured prompting
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])

            # Step 2: Create the chain
            chain = prompt | self.llm | StrOutputParser()

            # Step 3: Execute the chain ASYNCHRONOUSLY
            # This is the key difference - using ainvoke instead of invoke
            logger.debug("Sending async request to Ollama...")
            raw_response = await chain.ainvoke({})  # Non-blocking!

            logger.debug(f"Received async response ({len(raw_response)} chars)")

            # Step 4: Parse the response into structured SOAPNote
            soap_note = self._parse_soap_response(raw_response)

            # Step 5: Validate clinical safety and logic
            logger.debug("Validating SOAP note for safety and clinical logic...")

            safety_warnings = self._validate_clinical_safety(soap_note)
            logic_warnings = self._validate_clinical_logic(soap_note)

            all_warnings = safety_warnings + logic_warnings

            if all_warnings:
                logger.warning(f"SOAP note validation found {len(all_warnings)} issue(s)")

                warning_text = "\n\n" + "="*70 + "\n"
                warning_text += "VALIDATION WARNINGS\n"
                warning_text += "="*70 + "\n"
                warning_text += "\n".join(all_warnings)
                warning_text += "\n" + "="*70

                soap_note.plan = soap_note.plan + warning_text

                for warning in all_warnings:
                    logger.warning(warning)
            else:
                logger.info("SOAP note passed all validation checks")

            logger.info("Professional SOAP note generated successfully (async)")
            return soap_note

        except SOAPGenerationError:
            raise
        except Exception as e:
            logger.error(f"Async SOAP generation failed: {e}")
            raise SOAPGenerationError(
                reason=str(e),
                transcription_preview=transcription[:500]
            )
    
    def _parse_soap_response(self, response: str) -> SOAPNote:
        """
        Parse the LLM's text response into a structured SOAPNote.
        
        This is a crucial function that converts unstructured LLM output
        into our structured data model. We use regex patterns to extract
        each SOAP section.
        
        Parsing Strategy:
        1. Look for section headers (SUBJECTIVE:, OBJECTIVE:, etc.)
        2. Extract content between headers
        3. Skip empty sections (LLM sometimes outputs empty headers first)
        4. Clean and validate each section
        5. Create SOAPNote object
        
        Args:
            response: Raw text response from LLM
            
        Returns:
            Structured SOAPNote object
        """
        logger.debug("Parsing SOAP response...")
        
        # Define section patterns - match all occurrences
        # Using case-insensitive matching and flexible whitespace
        patterns = {
            'subjective': r'(?:\*\*)?SUBJECTIVE:(?:\*\*)?[\s\n]+(.*?)(?=(?:\*\*)?OBJECTIVE:(?:\*\*)?|---[\s\n]+(?:\*\*)?OBJECTIVE|$)',
            'objective': r'(?:\*\*)?OBJECTIVE:(?:\*\*)?[\s\n]+(.*?)(?=(?:\*\*)?ASSESSMENT:(?:\*\*)?|---[\s\n]+(?:\*\*)?ASSESSMENT|$)',
            'assessment': r'(?:\*\*)?ASSESSMENT:(?:\*\*)?[\s\n]+(.*?)(?=(?:\*\*)?PLAN:(?:\*\*)?|---[\s\n]+(?:\*\*)?PLAN|$)',
            'plan': r'(?:\*\*)?PLAN:(?:\*\*)?[\s\n]+(.*?)(?=---|End of SOAP Note|VALIDATION WARNINGS|$)'
        }
        
        sections = {}
        
        for section_name, pattern in patterns.items():
            # Find ALL matches for this section
            matches = list(re.finditer(pattern, response, re.IGNORECASE | re.DOTALL))
            
            if matches:
                # Try each match and use the first one with meaningful content
                content = None
                for match in matches:
                    candidate = match.group(1).strip()
                    # Clean up the content
                    cleaned = self._clean_section_content(candidate)
                    # Use this match if it has substantial content (not just dashes/whitespace)
                    if len(cleaned) > 5 and cleaned not in ['---', 'Note**']:
                        content = cleaned
                        break
                
                if content:
                    sections[section_name] = content
                else:
                    logger.warning(f"Found {section_name} header but no meaningful content")
                    sections[section_name] = "Not documented in this encounter"
            else:
                logger.warning(f"Could not find {section_name} section in response")
                sections[section_name] = "Not documented in this encounter"
        
        # Validate we got meaningful content
        total_content = sum(len(s) for s in sections.values())
        if total_content < 50:  # Arbitrary minimum
            logger.warning("SOAP note content seems too short")
        
        return SOAPNote(
            subjective=sections['subjective'],
            objective=sections['objective'],
            assessment=sections['assessment'],
            plan=sections['plan']
        )
    
    def _clean_section_content(self, content: str) -> str:
        """
        Clean up section content from LLM response.
        
        This handles common issues:
        1. Extra whitespace
        2. Markdown artifacts
        3. Incomplete sentences
        4. Empty headers and separators
        5. Stray formatting markers
        """
        # Remove leading/trailing whitespace
        content = content.strip()
        
        # Remove leading dashes and Note** artifacts
        content = re.sub(r'^[-]+\s*', '', content)
        content = re.sub(r'^Note\*\*\s*', '', content)
        content = re.sub(r'^---\s*', '', content, flags=re.MULTILINE)
        
        # Remove markdown headers if present
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
        
        # Remove excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove any remaining section headers that got included (non-bolded versions)
        content = re.sub(
            r'^(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN|S|O|A|P):\s*$',
            '',
            content,
            flags=re.IGNORECASE | re.MULTILINE
        )
        
        # Remove standalone dashes on their own lines
        content = re.sub(r'^---+$', '', content, flags=re.MULTILINE)
        
        # Clean up multiple blank lines again after removals
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()

    def _validate_clinical_safety(self, soap_note: SOAPNote) -> list[str]:
        """
        Validate SOAP note for critical safety issues.

        Checks for:
        1. IPV/Abuse cases with unsafe safety planning
        2. Problematic safety plan language

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        # Check if this is an IPV/abuse case
        assessment_lower = soap_note.assessment.lower()
        plan_lower = soap_note.plan.lower()

        ipv_keywords = [
            'intimate partner violence', 'ipv', 'domestic violence',
            'domestic abuse', 'partner abuse', 'spousal abuse',
            'physical abuse', 'assault', 'violence by partner',
            'violence by spouse', 'relationship violence'
        ]

        is_ipv_case = any(keyword in assessment_lower for keyword in ipv_keywords)

        if is_ipv_case:
            logger.info("Detected IPV/abuse case - validating safety plan...")

            # Problematic phrases that should never appear in IPV safety plans
            unsafe_phrases = [
                'with partner', 'with spouse', 'with abuser',
                'include partner', 'include spouse',
                'discuss with partner', 'discuss with spouse',
                'involve partner', 'involve spouse',
                'partner in safety', 'spouse in safety'
            ]

            for phrase in unsafe_phrases:
                if phrase in plan_lower:
                    warnings.append(
                        f"⚠️  CRITICAL SAFETY ISSUE: IPV case contains unsafe phrase '{phrase}' in safety plan. "
                        f"Safety planning must NEVER include the abuser/partner. "
                        f"Patient safety may be compromised."
                    )
                    logger.error(f"Safety violation detected: '{phrase}' found in IPV safety plan")

            # Check if proper safety resources are mentioned
            safety_resources = [
                'hotline', 'shelter', 'crisis', 'emergency contact',
                'safe', 'confidential', 'resource'
            ]

            has_safety_resources = any(resource in plan_lower for resource in safety_resources)

            if not has_safety_resources:
                warnings.append(
                    "⚠️  WARNING: IPV case detected but no safety resources (hotline, shelter, crisis line) "
                    "mentioned in plan. Consider adding patient-centered safety resources."
                )

        # Check ICD-10 codes for common hallucinations
        icd_warnings = self._validate_icd10_codes(soap_note)
        warnings.extend(icd_warnings)

        return warnings

    def _validate_icd10_codes(self, soap_note: SOAPNote) -> list[str]:
        """
        Validate ICD-10 codes for common hallucinations and errors.

        Checks for:
        1. Wrong code categories (F32 for anxiety, X codes for medical diagnoses)
        2. Code format validation

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        assessment_lower = soap_note.assessment.lower()

        # Find ICD-10 codes in assessment (format: ICD-10: XXX.XX)
        icd_pattern = r'icd-10:\s*([A-Z]\d{2}(?:\.\d{1,2})?)'
        icd_matches = re.findall(icd_pattern, soap_note.assessment, re.IGNORECASE)

        if icd_matches:
            logger.debug(f"Found {len(icd_matches)} ICD-10 code(s): {icd_matches}")

            for code in icd_matches:
                code_upper = code.upper()
                logger.debug(f"Validating ICD-10 code: {code_upper}")

                # Check for common hallucinations
                # F32.X = Depression, NOT anxiety
                if code_upper.startswith('F32') and 'anxiety' in assessment_lower:
                    warnings.append(
                        f"⚠️  ICD-10 HALLUCINATION: Code {code_upper} is for Major Depressive Disorder, "
                        f"but diagnosis mentions 'anxiety'. F32.X codes are for depression, NOT anxiety. "
                        f"Anxiety disorders use F41.X codes. This is a common hallucination error."
                    )
                    logger.error(f"ICD-10 hallucination detected: {code_upper} for anxiety")

                # X codes (External causes) should not be primary diagnosis
                if code_upper.startswith('X') or code_upper.startswith('Y'):
                    if not ('external cause' in assessment_lower or 'secondary' in assessment_lower):
                        warnings.append(
                            f"⚠️  ICD-10 ERROR: Code {code_upper} is an External Cause code (accidents, assaults, events). "
                            f"These should NOT be used as primary diagnoses for medical conditions. "
                            f"Example hallucination: X34.0 (earthquake victim) for IPV."
                        )
                        logger.error(f"External cause code used as primary diagnosis: {code_upper}")

                # Check for mismatched mental health codes
                if code_upper.startswith('F41') and 'depress' in assessment_lower and 'anxiety' not in assessment_lower:
                    warnings.append(
                        f"⚠️  ICD-10 MISMATCH: Code {code_upper} is for anxiety disorders, "
                        f"but diagnosis primarily mentions depression. Consider F32.X codes instead."
                    )

        return warnings

    def _validate_clinical_logic(self, soap_note: SOAPNote) -> list[str]:
        """
        Validate SOAP note for clinical logic issues.

        Checks for:
        1. Medications prescribed without documented findings
        2. Vital signs in ROS (should be in Objective)
        3. Clinical inconsistencies

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        # Check for vital signs in ROS (common error from problematic output)
        subjective_lower = soap_note.subjective.lower()

        # Log what we're checking for debugging
        logger.debug(f"Validating clinical logic. Subjective text length: {len(subjective_lower)} chars")
        logger.debug(f"First 200 chars of subjective (lowercase): {subjective_lower[:200]}")

        # Vital sign patterns that should NEVER appear in Subjective/ROS
        vital_patterns = [
            r'bp\s*[:=]?\s*\d+[/]\d+',  # BP 120/80
            r'blood pressure\s*[:=]?\s*\d+[/]\d+',
            r'hr\s*[:=]?\s*\d+',  # HR 72
            r'heart rate\s*[:=]?\s*\d+',
            r'rr\s*[:=]?\s*\d+',  # RR 12
            r'respiratory rate\s*[:=]?\s*\d+',
            r'temperature\s*[:=]?\s*\d+',
            r't\s*[:=]?\s*\d+\.?\d*\s*°?[fc]',  # T 98.6°F
            r'o2\s*sat\s*[:=]?\s*\d+%?',  # O2 sat 98%
        ]

        for pattern in vital_patterns:
            match = re.search(pattern, subjective_lower)
            if match:
                logger.debug(f"Vital sign pattern '{pattern}' matched: '{match.group()}'")
                warnings.append(
                    f"⚠️  STRUCTURE ERROR: Vital sign measurement found in SUBJECTIVE section. "
                    f"All vital signs (BP, HR, RR, T, O2 sat) must be in OBJECTIVE section only. "
                    f"ROS should contain patient-reported symptoms, not measured values."
                )
                logger.warning(f"Vital signs detected in Subjective section: '{match.group()}'")
                break  # Only warn once for this issue

        # Check for pain medications without documented pain
        plan_lower = soap_note.plan.lower()
        objective_lower = soap_note.objective.lower()

        pain_meds = [
            'ibuprofen', 'acetaminophen', 'naproxen', 'aspirin',
            'tylenol', 'advil', 'motrin', 'aleve',
            'oxycodone', 'hydrocodone', 'morphine', 'tramadol'
        ]

        has_pain_med = any(med in plan_lower for med in pain_meds)

        if has_pain_med:
            # Check if pain is documented in subjective or objective
            pain_keywords = [
                'pain', 'tender', 'ache', 'sore', 'discomfort',
                'hurts', 'painful'
            ]

            has_pain_documented = (
                any(keyword in subjective_lower for keyword in pain_keywords) or
                any(keyword in objective_lower for keyword in pain_keywords)
            )

            if not has_pain_documented:
                warnings.append(
                    "⚠️  CLINICAL LOGIC ERROR: Pain medication prescribed but no pain documented "
                    "in Subjective or Objective sections. Treatment must be supported by documented findings."
                )
                logger.warning("Pain medication prescribed without documented pain")

        # Check for "no significant findings" + medications
        if 'no significant findings' in objective_lower:
            # Check if any medications are prescribed (excluding vitamins, preventive meds)
            medication_section = re.search(r'medications?[:]\s*(.*?)(?=\n\d+\.|\n[A-Z]|$)',
                                          plan_lower, re.DOTALL)

            if medication_section:
                meds_text = medication_section.group(1)
                # Exclude preventive/maintenance meds
                exclude_terms = ['vitamin', 'supplement', 'follow-up', 'continue']

                if meds_text and not any(term in meds_text for term in exclude_terms):
                    if len(meds_text.strip()) > 10:  # Has actual medication content
                        warnings.append(
                            "⚠️  CLINICAL LOGIC WARNING: Objective section states 'no significant findings' "
                            "but medications are prescribed in Plan. Ensure findings support treatment."
                        )

        # Check for TID/BID/QID + PRN conflicts (contradictory)
        # Pattern: looks for scheduled frequency (TID/BID/QID) followed by PRN/as needed
        tid_prn_patterns = [
            r'(tid|bid|qid)\s+(prn|as\s+needed)',  # TID PRN
            r'(prn|as\s+needed)\s+(tid|bid|qid)',  # PRN TID
            r'(tid|bid|qid).*\s+(prn|as\s+needed)',  # TID ... PRN (with words between)
        ]

        for pattern in tid_prn_patterns:
            match = re.search(pattern, plan_lower, re.IGNORECASE)
            if match:
                warnings.append(
                    f"⚠️  MEDICATION ERROR: Found contradictory frequency '{match.group()}'. "
                    f"TID/BID/QID means scheduled (at set times). PRN/as needed means when patient decides. "
                    f"Cannot be both! Choose one: either scheduled (TID/BID/QID) OR as needed (PRN)."
                )
                logger.warning(f"TID+PRN conflict detected: '{match.group()}'")
                break  # Only warn once

        # Check for high-dose benzodiazepines
        # Clonazepam >2mg/day, Lorazepam >4mg/day, Alprazolam >4mg/day
        benzo_patterns = [
            (r'clonazepam\s+(\d+(?:\.\d+)?)\s*mg', 2.0, 'clonazepam'),
            (r'klonopin\s+(\d+(?:\.\d+)?)\s*mg', 2.0, 'klonopin (clonazepam)'),
            (r'lorazepam\s+(\d+(?:\.\d+)?)\s*mg', 4.0, 'lorazepam'),
            (r'ativan\s+(\d+(?:\.\d+)?)\s*mg', 4.0, 'ativan (lorazepam)'),
            (r'alprazolam\s+(\d+(?:\.\d+)?)\s*mg', 4.0, 'alprazolam'),
            (r'xanax\s+(\d+(?:\.\d+)?)\s*mg', 4.0, 'xanax (alprazolam)'),
        ]

        for pattern, max_dose, med_name in benzo_patterns:
            matches = re.findall(pattern, plan_lower, re.IGNORECASE)
            if matches:
                for dose_str in matches:
                    try:
                        dose = float(dose_str)
                        # Check if it's TID/QID (multiply by frequency)
                        if 'tid' in plan_lower:
                            total_daily = dose * 3
                        elif 'qid' in plan_lower:
                            total_daily = dose * 4
                        elif 'bid' in plan_lower:
                            total_daily = dose * 2
                        else:
                            total_daily = dose

                        if total_daily > max_dose:
                            warnings.append(
                                f"⚠️  HIGH-DOSE BENZODIAZEPINE: {med_name} {total_daily}mg/day exceeds "
                                f"typical maximum of {max_dose}mg/day. High doses require clear justification "
                                f"and should be approached cautiously due to dependence risk."
                            )
                            logger.warning(f"High benzo dose: {med_name} {total_daily}mg/day")
                    except ValueError:
                        pass  # Skip if dose can't be converted to float

        return warnings


class MockSOAPGenerator:
    """
    Mock generator for testing.
    
    This allows testing the pipeline without a real LLM.
    Supports both sync and async interfaces for comprehensive testing.
    """
    
    def __init__(self, mock_note: Optional[SOAPNote] = None):
        self.mock_note = mock_note or SOAPNote(
            subjective="Mock subjective content",
            objective="Mock objective content", 
            assessment="Mock assessment content",
            plan="Mock plan content"
        )
        self.call_count = 0
    
    def generate(self, transcription: str, language: str = "en") -> SOAPNote:
        """Return mock SOAP note (sync)."""
        self.call_count += 1
        return self.mock_note
    
    async def agenerate(self, transcription: str, language: str = "en") -> SOAPNote:
        """Return mock SOAP note (async)."""
        self.call_count += 1
        # Simulate some async delay for realistic testing
        await asyncio.sleep(0.01)
        return self.mock_note


# =============================================================================
# Factory Function
# =============================================================================

def create_soap_generator(
    settings: Optional[Settings] = None,
    use_mock: bool = False,
    mock_note: Optional[SOAPNote] = None
) -> SOAPGeneratorProtocol:
    """
    Factory function to create the appropriate SOAP generator.
    
    Args:
        settings: Application settings
        use_mock: If True, returns a mock generator
        mock_note: SOAPNote to return from mock generator
        
    Returns:
        A SOAP generator instance
    """
    if use_mock:
        logger.info("Creating mock SOAP generator")
        return MockSOAPGenerator(mock_note=mock_note)
    
    logger.info("Creating Ollama SOAP generator")
    return OllamaSOAPGenerator(settings=settings)
