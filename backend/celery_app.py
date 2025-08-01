"""
Celery configuration for async task processing
"""

import os
from celery import Celery
from kombu import Exchange, Queue

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "datasetgen",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.pipeline_tasks", "tasks.processing_tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    
    # Result backend settings
    result_expires=86400,  # 24 hours
    
    # Task routing
    task_routes={
        "tasks.pipeline_tasks.*": {"queue": "pipeline"},
        "tasks.processing_tasks.*": {"queue": "processing"},
    },
    
    # Queue configuration
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("pipeline", Exchange("pipeline"), routing_key="pipeline"),
        Queue("processing", Exchange("processing"), routing_key="processing"),
    ),
    
    # Task time limits
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,  # 2 hour hard limit
    
    # Task result tracking
    task_track_started=True,
    task_send_sent_event=True,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-old-files": {
        "task": "tasks.maintenance.cleanup_old_files",
        "schedule": 86400.0,  # Daily
    },
    "update-metrics": {
        "task": "tasks.maintenance.update_metrics",
        "schedule": 300.0,  # Every 5 minutes
    },
}