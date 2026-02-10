"""
WebSocket Endpoints
===================

Real-time job progress streaming via WebSocket.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
import redis.asyncio as aioredis
import json
import asyncio
from typing import Optional

from config import get_settings
from api.services.job_manager import JobManager

router = APIRouter()


@router.websocket("/jobs/{job_id}/stream")
async def websocket_job_stream(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates.

    Connect to receive live updates as the job progresses.

    Args:
        websocket: FastAPI WebSocket connection
        job_id: Unique job identifier

    Usage:
        ```javascript
        const ws = new WebSocket('ws://localhost:8000/api/v1/jobs/{job_id}/stream');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(`Progress: ${data.progress}% - ${data.current_stage}`);

            if (data.status === 'completed') {
                console.log('Job completed!', data.result);
            } else if (data.status === 'failed') {
                console.error('Job failed!', data.error);
            }
        };
        ```
    """
    await websocket.accept()

    settings = get_settings()
    job_manager = JobManager()
    redis_client: Optional[aioredis.Redis] = None
    pubsub = None

    try:
        # Verify job exists
        job_data = job_manager.get_job(job_id)
        if not job_data:
            await websocket.send_json({
                "error": "Job not found",
                "job_id": job_id,
                "message": f"No job found with ID: {job_id}"
            })
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Send initial status
        await websocket.send_json(job_data)

        # If job is already completed or failed, close connection
        if job_data.get("status") in ["completed", "failed"]:
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
            return

        # Subscribe to Redis pub/sub for updates
        redis_client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True
        )

        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"job_updates:{job_id}")

        # Listen for updates
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    # Parse and send update to client
                    data = json.loads(message["data"])
                    await websocket.send_json(data)

                    # Close connection if job is completed or failed
                    if data.get("status") in ["completed", "failed"]:
                        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                        break

                except json.JSONDecodeError as e:
                    # Log error but continue listening
                    await websocket.send_json({
                        "error": "Invalid message format",
                        "message": f"Failed to parse update: {str(e)}"
                    })
                except Exception as e:
                    # Send error to client but continue
                    await websocket.send_json({
                        "error": "Update processing error",
                        "message": str(e)
                    })

    except WebSocketDisconnect:
        # Client disconnected - clean up silently
        pass

    except Exception as e:
        # Unexpected error - try to send error message before closing
        try:
            await websocket.send_json({
                "error": "Internal server error",
                "message": f"An unexpected error occurred: {str(e)}"
            })
        except:
            pass  # Connection might already be closed

    finally:
        # Cleanup Redis resources
        if pubsub:
            try:
                await pubsub.unsubscribe(f"job_updates:{job_id}")
                await pubsub.close()
            except:
                pass  # Best effort cleanup

        if redis_client:
            try:
                await redis_client.close()
            except:
                pass  # Best effort cleanup

        # Ensure WebSocket is closed
        try:
            await websocket.close()
        except:
            pass  # Connection might already be closed
