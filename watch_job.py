#!/usr/bin/env python3
"""
Watch Job Progress via WebSocket
=================================

Simple script to watch a specific job's progress in real-time via WebSocket.

Usage:
    # Upload audio via FastAPI docs (http://localhost:8000/api/docs)
    # Copy the job_id from response, then:
    python watch_job.py <job_id>

Example:
    python watch_job.py 123e4567-e89b-12d3-a456-426614174000
"""

import asyncio
import json
import sys
import websockets
from datetime import datetime


# Configuration
WS_BASE_URL = "ws://localhost:8000/api/v1"


def print_colored(message: str, color: str = "default"):
    """Print colored output."""
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "reset": "\033[0m",
        "default": ""
    }

    reset = colors["reset"]
    color_code = colors.get(color, "")
    print(f"{color_code}{message}{reset}")


async def watch_job(job_id: str):
    """Watch a job's progress via WebSocket."""
    ws_url = f"{WS_BASE_URL}/jobs/{job_id}/stream"

    print_colored(f"\n{'='*70}", "cyan")
    print_colored(f"  🔍 Watching Job: {job_id}", "cyan")
    print_colored(f"{'='*70}\n", "cyan")

    print_colored(f"Connecting to: {ws_url}", "blue")

    try:
        async with websockets.connect(ws_url) as websocket:
            print_colored("✓ WebSocket connected!\n", "green")

            # Track timing
            start_time = datetime.now()
            last_progress = -1

            # Listen for messages
            async for message in websocket:
                data = json.loads(message)

                # Check for errors (only if error has a value, not just None)
                if data.get("error"):
                    print_colored(f"\n❌ Error: {data.get('message', data['error'])}", "red")
                    break

                # Extract data
                status = data.get("status", "unknown")
                progress = data.get("progress", 0)
                stage = data.get("current_stage", "N/A")

                # Only show updates when progress changes
                if progress != last_progress:
                    elapsed = (datetime.now() - start_time).total_seconds()

                    # Progress bar
                    bar_length = 40
                    filled = int(bar_length * progress / 100)
                    bar = "█" * filled + "░" * (bar_length - filled)

                    # Color based on status
                    bar_color = "blue"
                    if status == "completed":
                        bar_color = "green"
                    elif status == "failed":
                        bar_color = "red"
                    elif status == "processing":
                        bar_color = "yellow"

                    # Print progress
                    status_text = f"[{bar}] {progress}% | {stage}"
                    print_colored(f"\r{status_text} ({elapsed:.1f}s)", bar_color)

                    last_progress = progress

                # Handle completion
                if status == "completed":
                    print_colored("\n\n✓ Job completed successfully!", "green")

                    result = data.get("result")
                    if result:
                        print_colored(f"\n{'='*70}", "green")
                        print_colored("  📋 RESULT", "green")
                        print_colored(f"{'='*70}\n", "green")

                        # Display transcription if available
                        if "transcription" in result:
                            print_colored("Transcription:", "cyan")
                            print(result["transcription"][:500] + "..." if len(result["transcription"]) > 500 else result["transcription"])
                            print()

                        # Display SOAP note if available
                        if "soap_note" in result:
                            soap = result["soap_note"]

                            print_colored("SOAP Note:", "cyan")
                            print_colored("-" * 70, "cyan")

                            if soap.get("subjective"):
                                print_colored("\n📌 Subjective:", "magenta")
                                print(soap["subjective"])

                            if soap.get("objective"):
                                print_colored("\n📊 Objective:", "magenta")
                                print(soap["objective"])

                            if soap.get("assessment"):
                                print_colored("\n🔍 Assessment:", "magenta")
                                print(soap["assessment"])

                            if soap.get("plan"):
                                print_colored("\n📝 Plan:", "magenta")
                                print(soap["plan"])

                            print_colored("\n" + "-" * 70, "cyan")

                        # Display metadata if available
                        if "metadata" in result:
                            meta = result["metadata"]
                            print_colored("\nMetadata:", "cyan")
                            print(f"  Duration: {meta.get('duration_seconds', 'N/A')}s")
                            print(f"  Processing time: {meta.get('processing_time_seconds', 'N/A')}s")
                            if meta.get("language"):
                                print(f"  Language: {meta['language']}")

                    print_colored(f"\n{'='*70}\n", "green")
                    break

                elif status == "failed":
                    print_colored("\n\n❌ Job failed!", "red")
                    error = data.get("error", {})
                    print_colored(f"Error: {error}", "red")
                    break

            print_colored("Connection closed.", "blue")

    except websockets.exceptions.InvalidStatusCode as e:
        print_colored(f"\n❌ WebSocket connection failed: {e}", "red")

        if e.status_code == 404:
            print_colored("\nPossible reasons:", "yellow")
            print_colored("  1. Job ID not found (may have expired - 24hr TTL)", "yellow")
            print_colored("  2. Job ID is incorrect", "yellow")
            print_colored(f"\nJob ID provided: {job_id}", "yellow")
        elif e.status_code == 403:
            print_colored("\n❌ Access denied to job", "red")
        else:
            print_colored(f"\nHTTP Status Code: {e.status_code}", "red")

    except websockets.exceptions.WebSocketException as e:
        print_colored(f"\n❌ WebSocket error: {e}", "red")

    except Exception as e:
        print_colored(f"\n❌ Unexpected error: {e}", "red")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_colored("\n❌ Usage: python watch_job.py <job_id>\n", "red")
        print_colored("Example:", "blue")
        print_colored("  python watch_job.py 123e4567-e89b-12d3-a456-426614174000\n", "blue")
        print_colored("How to get a job ID:", "cyan")
        print_colored("  1. Go to http://localhost:8000/api/docs", "cyan")
        print_colored("  2. Use POST /api/v1/process to upload an audio file", "cyan")
        print_colored("  3. Copy the 'job_id' from the response", "cyan")
        print_colored("  4. Run: python watch_job.py <job_id>\n", "cyan")
        sys.exit(1)

    job_id = sys.argv[1].strip()

    if not job_id:
        print_colored("\n❌ Error: Job ID cannot be empty\n", "red")
        sys.exit(1)

    try:
        asyncio.run(watch_job(job_id))
    except KeyboardInterrupt:
        print_colored("\n\n⚠️  Interrupted by user", "yellow")
        sys.exit(0)


if __name__ == "__main__":
    main()
