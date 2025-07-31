#!/usr/bin/env python
"""
Celery worker entry point

Run with:
    celery -A celery_worker worker --loglevel=info
    
For development with auto-reload:
    watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- celery -A celery_worker worker --loglevel=info
"""

import os
import sys

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Celery app
from services.pipeline.celery_tasks import celery_app

# Import all tasks to register them
from services.pipeline import celery_tasks

if __name__ == '__main__':
    celery_app.start()