"""
Command Line Interface for DocuMed AI
=====================================

This module provides the command-line interface for processing medical
audio files into SOAP notes.

Usage:
------
    # Process a single file
    python -m documed_core audio.mp3
    
    # Process with options
    python -m documed_core audio.mp3 --output ./results --verbose
    
    # Transcription only (no SOAP generation)
    python -m documed_core audio.mp3 --transcribe-only
    
    # Process text directly
    python -m documed_core --text "Patient presents with..."

CLI Design Principles:
---------------------
1. Sensible defaults (works out of the box)
2. Clear help messages
3. Progress feedback
4. Exit codes for scripting
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from pipeline import MedicalDocumentationPipeline, save_result_to_file
from models import ProcessingStatus
from config import get_settings
from exceptions import DocuMedError


# ANSI colors for terminal output
class Colors:
    """ANSI color codes for pretty terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def colorize(text: str, color: str) -> str:
    """Add color to text if terminal supports it."""
    # Check if stdout is a terminal
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text


def print_banner():
    """Print a nice banner for the CLI."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     ██████╗  ██████╗  ██████╗██╗   ██╗███╗   ███╗███████╗   ║
    ║     ██╔══██╗██╔═══██╗██╔════╝██║   ██║████╗ ████║██╔════╝   ║
    ║     ██║  ██║██║   ██║██║     ██║   ██║██╔████╔██║█████╗     ║
    ║     ██║  ██║██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══╝     ║
    ║     ██████╔╝╚██████╔╝╚██████╗╚██████╔╝██║ ╚═╝ ██║███████╗   ║
    ║     ╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ║
    ║                                                              ║
    ║           Medical Audio → SOAP Notes                         ║
    ║           Free • Local • Private                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(colorize(banner, Colors.CYAN))


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    This defines all CLI options and their help text.
    """
    parser = argparse.ArgumentParser(
        prog="documed",
        description="Convert medical audio consultations to SOAP notes",
        epilog="Example: python -m documed_core consultation.mp3 --output ./notes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional argument: audio file
    parser.add_argument(
        "audio_file",
        nargs="?",  # Optional (can use --text instead)
        help="Path to the audio file to process"
    )
    
    # Alternative input: direct text
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Process text directly instead of audio file"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files, just print"
    )
    
    # Processing options
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only transcribe audio, don't generate SOAP note"
    )
    
    parser.add_argument(
        "--soap-only",
        action="store_true",
        help="Only generate SOAP from provided text (requires --text)"
    )
    
    # Model options
    parser.add_argument(
        "--whisper-model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (overrides config)"
    )
    
    parser.add_argument(
        "--ollama-model",
        type=str,
        help="Ollama model name (overrides config)"
    )
    
    # Output format
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with debug info"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Don't show the banner"
    )
    
    return parser


def setup_logging_for_cli(verbose: bool, quiet: bool) -> None:
    """Configure logging based on CLI flags."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )


def progress_callback(status: ProcessingStatus, message: str) -> None:
    """
    Callback for progress updates.
    
    This is called by the pipeline at each stage.
    """
    status_colors = {
        ProcessingStatus.PENDING: Colors.YELLOW,
        ProcessingStatus.TRANSCRIBING: Colors.BLUE,
        ProcessingStatus.GENERATING: Colors.CYAN,
        ProcessingStatus.COMPLETED: Colors.GREEN,
        ProcessingStatus.FAILED: Colors.RED,
    }
    
    color = status_colors.get(status, Colors.ENDC)
    status_str = f"[{status.value.upper():^12}]"
    
    print(f"{colorize(status_str, color)} {message}")


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    setup_logging_for_cli(parsed_args.verbose, parsed_args.quiet)
    
    # Show banner unless suppressed
    if not parsed_args.no_banner and not parsed_args.quiet:
        print_banner()
    
    # Validate input
    if not parsed_args.audio_file and not parsed_args.text:
        parser.error("Either audio_file or --text is required")
    
    if parsed_args.soap_only and not parsed_args.text:
        parser.error("--soap-only requires --text")
    
    try:
        # Apply CLI overrides BEFORE loading settings
        # This is important because get_settings() uses @lru_cache
        import os
        if parsed_args.whisper_model:
            os.environ["DOCUMED_WHISPER_MODEL"] = parsed_args.whisper_model
        if parsed_args.ollama_model:
            os.environ["DOCUMED_OLLAMA_MODEL"] = parsed_args.ollama_model
        
        # Build settings with any overrides
        settings = get_settings()
        
        # Create pipeline
        pipeline = MedicalDocumentationPipeline()
        
        # Process based on mode
        if parsed_args.soap_only:
            # SOAP only from text
            if not parsed_args.quiet:
                print(colorize("\n📝 Generating SOAP note from text...\n", Colors.CYAN))
            
            soap_note = pipeline.generate_soap_only(parsed_args.text)
            
            if parsed_args.json:
                import json
                print(json.dumps(soap_note.model_dump(), indent=2))
            else:
                print(soap_note.to_formatted_string())
        
        elif parsed_args.transcribe_only:
            # Transcription only
            if not parsed_args.quiet:
                print(colorize("\n🎤 Transcribing audio...\n", Colors.CYAN))
            
            result = pipeline.transcribe_only(parsed_args.audio_file)
            
            if parsed_args.json:
                import json
                print(json.dumps(result.model_dump(), indent=2))
            else:
                print(colorize("\n─── TRANSCRIPTION ───\n", Colors.HEADER))
                print(result.text)
                print(colorize(f"\n[Duration: {result.duration_seconds:.1f}s | Language: {result.language}]", Colors.CYAN))
        
        elif parsed_args.text:
            # Process text input (no audio file)
            if not parsed_args.quiet:
                print(colorize("\n📝 Processing text input...\n", Colors.CYAN))
            
            soap_note = pipeline.generate_soap_only(parsed_args.text)
            
            if parsed_args.json:
                import json
                print(json.dumps(soap_note.model_dump(), indent=2))
            else:
                print(soap_note.to_formatted_string())
        
        else:
            # Full pipeline: audio → SOAP
            if not parsed_args.quiet:
                print(colorize(f"\n🎵 Processing: {parsed_args.audio_file}\n", Colors.CYAN))
            
            # Use progress callback unless quiet
            callback = None if parsed_args.quiet else progress_callback
            
            result = pipeline.process(
                parsed_args.audio_file,
                progress_callback=callback
            )
            
            # Check for errors
            if result.status == ProcessingStatus.FAILED:
                print(colorize(f"\n❌ Error: {result.error_message}", Colors.RED))
                return 1
            
            # Output results
            if parsed_args.json:
                import json
                print(json.dumps(result.model_dump(mode='json'), indent=2, default=str))
            else:
                print(result.soap_note.to_formatted_string())
            
            # Save if requested
            if not parsed_args.no_save:
                saved = save_result_to_file(result, parsed_args.output)
                if not parsed_args.quiet:
                    print(colorize(f"\n💾 Results saved to: {parsed_args.output}", Colors.GREEN))
                    for file_type, path in saved.items():
                        print(f"   • {file_type}: {path}")
        
        if not parsed_args.quiet:
            print(colorize("\n✅ Done!\n", Colors.GREEN))
        
        return 0
        
    except DocuMedError as e:
        print(colorize(f"\n❌ Error: {e.message}", Colors.RED))
        if parsed_args.verbose and e.details:
            print(colorize(f"   Details: {e.details}", Colors.YELLOW))
        return 1
        
    except KeyboardInterrupt:
        print(colorize("\n\n⚠️  Interrupted by user", Colors.YELLOW))
        return 130
        
    except Exception as e:
        print(colorize(f"\n❌ Unexpected error: {e}", Colors.RED))
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
