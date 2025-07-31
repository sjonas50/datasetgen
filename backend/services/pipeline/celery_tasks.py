from celery import Celery, Task
from celery.result import AsyncResult
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

from core.config import settings
from core.logging import get_logger
from .executor import PipelineExecutor
from .base import PipelineConfig, PipelineExecution

logger = get_logger(__name__)

# Create Celery app
celery_app = Celery(
    'pipeline_tasks',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['services.pipeline.celery_tasks']
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 min soft limit
    worker_prefetch_multiplier=1,  # For fair task distribution
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
)


class PipelineTask(Task):
    """Base task class with error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Pipeline task {task_id} failed: {str(exc)}")
        super().on_failure(exc, task_id, args, kwargs, einfo)
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Pipeline task {task_id} retrying: {str(exc)}")
        super().on_retry(exc, task_id, args, kwargs, einfo)


@celery_app.task(
    bind=True,
    base=PipelineTask,
    name='pipeline.execute',
    max_retries=3,
    default_retry_delay=60
)
def execute_pipeline_task(
    self,
    pipeline_config: Dict[str, Any],
    initial_data: Optional[Dict[str, Any]] = None,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a pipeline asynchronously via Celery
    
    Args:
        pipeline_config: Pipeline configuration dict
        initial_data: Initial data for the pipeline
        resume_from: Execution ID to resume from
    
    Returns:
        Execution result dict
    """
    try:
        # Convert dict to PipelineConfig
        config = PipelineConfig(**pipeline_config)
        
        # Create executor
        executor = PipelineExecutor()
        
        # Run async execution in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Update task state
            self.update_state(
                state='RUNNING',
                meta={
                    'pipeline_name': config.name,
                    'status': 'initializing',
                    'started_at': datetime.utcnow().isoformat()
                }
            )
            
            # Execute pipeline
            execution = loop.run_until_complete(
                executor.execute(config, initial_data, resume_from)
            )
            
            # Convert execution to dict for serialization
            result = {
                'execution_id': execution.id,
                'status': execution.status,
                'completed_steps': execution.completed_steps,
                'total_steps': execution.total_steps,
                'total_execution_time': execution.total_execution_time,
                'total_llm_cost': execution.total_llm_cost,
                'total_rows_processed': execution.total_rows_processed,
                'error': execution.error
            }
            
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        
        # Retry if possible
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        # Return error result
        return {
            'execution_id': None,
            'status': 'failed',
            'error': str(e)
        }


@celery_app.task(
    bind=True,
    name='pipeline.execute_step',
    max_retries=5,
    default_retry_delay=30
)
def execute_step_task(
    self,
    step_config: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a single pipeline step
    
    This allows for fine-grained distributed execution
    """
    from .base import StepConfig
    from .step_registry import StepRegistry
    
    try:
        # Convert dict to StepConfig
        config = StepConfig(**step_config)
        
        # Create step instance
        step = StepRegistry.create_step(config)
        
        # Run async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                step.execute(input_data, context or {})
            )
            
            return result.dict()
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Step execution failed: {str(e)}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        raise


@celery_app.task(name='pipeline.monitor')
def monitor_pipeline_task(execution_id: str) -> Dict[str, Any]:
    """
    Monitor a running pipeline execution
    
    Returns current status and progress
    """
    from .state_manager import PipelineStateManager
    
    state_manager = PipelineStateManager()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        execution = loop.run_until_complete(
            state_manager.load_state(execution_id)
        )
        
        if not execution:
            return {'error': f'Execution {execution_id} not found'}
        
        return {
            'execution_id': execution.id,
            'pipeline_name': execution.pipeline_config.name,
            'status': execution.status,
            'progress': f"{execution.completed_steps}/{execution.total_steps}",
            'current_step': execution.current_step,
            'started_at': execution.started_at.isoformat() if execution.started_at else None,
            'total_llm_cost': execution.total_llm_cost,
            'total_rows_processed': execution.total_rows_processed
        }
        
    finally:
        loop.close()


@celery_app.task(name='pipeline.cleanup')
def cleanup_old_executions_task(days_old: int = 7) -> Dict[str, Any]:
    """
    Clean up old pipeline executions
    
    Runs periodically to remove old state files
    """
    from .state_manager import PipelineStateManager
    from datetime import timedelta
    
    state_manager = PipelineStateManager()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        executions = loop.run_until_complete(
            state_manager.list_executions()
        )
        
        cleaned = 0
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        for execution in executions:
            if execution.get('completed_at'):
                completed_at = datetime.fromisoformat(execution['completed_at'])
                if completed_at < cutoff_date:
                    loop.run_until_complete(
                        state_manager.cleanup_execution(execution['id'])
                    )
                    cleaned += 1
        
        return {
            'cleaned': cleaned,
            'total_executions': len(executions)
        }
        
    finally:
        loop.close()


# Celery Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    'cleanup-old-executions': {
        'task': 'pipeline.cleanup',
        'schedule': 86400.0,  # Daily
        'args': (7,)  # Clean up executions older than 7 days
    },
}


class CeleryPipelineService:
    """Service for managing pipeline execution via Celery"""
    
    @staticmethod
    def submit_pipeline(
        config: PipelineConfig,
        initial_data: Optional[Any] = None,
        resume_from: Optional[str] = None
    ) -> str:
        """
        Submit a pipeline for async execution
        
        Returns:
            Task ID for tracking
        """
        task = execute_pipeline_task.delay(
            config.dict(),
            initial_data,
            resume_from
        )
        
        logger.info(f"Submitted pipeline {config.name} as task {task.id}")
        return task.id
    
    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """Get status of a pipeline task"""
        result = AsyncResult(task_id, app=celery_app)
        
        return {
            'task_id': task_id,
            'status': result.status,
            'ready': result.ready(),
            'successful': result.successful() if result.ready() else None,
            'result': result.result if result.ready() else None,
            'info': result.info
        }
    
    @staticmethod
    def cancel_pipeline(task_id: str) -> bool:
        """Cancel a running pipeline"""
        result = AsyncResult(task_id, app=celery_app)
        result.revoke(terminate=True)
        
        logger.info(f"Cancelled pipeline task {task_id}")
        return True
    
    @staticmethod
    def get_worker_stats() -> Dict[str, Any]:
        """Get Celery worker statistics"""
        inspect = celery_app.control.inspect()
        
        return {
            'active_tasks': inspect.active(),
            'scheduled_tasks': inspect.scheduled(),
            'reserved_tasks': inspect.reserved(),
            'stats': inspect.stats()
        }