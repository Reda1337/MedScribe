"""
Job Manager Service
===================

Redis-based job queue and status management.
Stub implementation for Phase 2, fully implemented in Phase 5.
"""

import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from config import get_settings


class JobManager:
    """
    Manage job queue and status using Redis.
    
    This is a stub implementation for Phase 2 that uses in-memory storage.
    Phase 5 will add full Redis support for persistence and pub/sub.
    """
    
    def __init__(self):
        """
        Initialize the job manager.
        
        In Phase 2: Uses in-memory dict (non-persistent)
        In Phase 5: Will use Redis for persistence
        """
        self.settings = get_settings()
        self._jobs: Dict[str, Dict[str, Any]] = {}  # In-memory storage
        self.job_ttl = 86400  # 24 hours
        
        # Try to connect to Redis (optional in Phase 2)
        self.redis_client = None
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                password=self.settings.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=2
            )
            # Test connection
            self.redis_client.ping()
        except Exception:
            # Redis not available - use in-memory fallback
            self.redis_client = None
    
    def create_job(self, job_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new job and store it.
        
        Args:
            job_type: Type of job (process, transcribe, generate_soap)
            metadata: Additional job metadata
            
        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "pending",
            "progress": 0,
            "current_stage": None,
            "result": None,
            "error": None,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if self.redis_client:
            # Store in Redis with TTL
            self.redis_client.setex(
                f"job:{job_id}",
                self.job_ttl,
                json.dumps(job_data)
            )
        else:
            # Store in memory
            self._jobs[job_id] = job_data
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job data by ID.
        
        Args:
            job_id: The job identifier
            
        Returns:
            Job data dict or None if not found
        """
        if self.redis_client:
            data = self.redis_client.get(f"job:{job_id}")
            if data:
                return json.loads(data)
            return None
        else:
            return self._jobs.get(job_id)
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        """
        Update job status/progress.
        
        Args:
            job_id: The job identifier
            updates: Dict of fields to update
        """
        job_data = self.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")
        
        job_data.update(updates)
        job_data["updated_at"] = datetime.utcnow().isoformat()
        
        if self.redis_client:
            self.redis_client.setex(
                f"job:{job_id}",
                self.job_ttl,
                json.dumps(job_data)
            )
            # Publish update for WebSocket subscribers (Phase 6)
            self.redis_client.publish(
                f"job_updates:{job_id}",
                json.dumps(job_data)
            )
        else:
            self._jobs[job_id] = job_data
    
    def set_job_progress(self, job_id: str, progress: int, stage: str) -> None:
        """
        Update job progress.
        
        Args:
            job_id: The job identifier
            progress: Progress percentage (0-100)
            stage: Current processing stage
        """
        self.update_job(job_id, {
            "status": "processing",
            "progress": progress,
            "current_stage": stage
        })
    
    def set_job_completed(self, job_id: str, result: Any) -> None:
        """
        Mark job as completed.
        
        Args:
            job_id: The job identifier
            result: The job result data
        """
        self.update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "current_stage": "completed",
            "result": result
        })
    
    def set_job_failed(self, job_id: str, error: Dict[str, Any]) -> None:
        """
        Mark job as failed.
        
        Args:
            job_id: The job identifier
            error: Error information dict
        """
        self.update_job(job_id, {
            "status": "failed",
            "error": error
        })
