from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import json
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics to track"""
    PIPELINE_EXECUTION = "pipeline_execution"
    STEP_EXECUTION = "step_execution"
    LLM_USAGE = "llm_usage"
    DATA_PROCESSING = "data_processing"
    ERROR_RATE = "error_rate"
    COST_TRACKING = "cost_tracking"


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics"""
    pipeline_id: str
    pipeline_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    rows_processed: int = 0
    execution_time_seconds: float = 0.0
    llm_tokens_used: int = 0
    llm_cost_usd: float = 0.0
    compute_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class StepMetrics:
    """Individual step execution metrics"""
    step_name: str
    step_type: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    rows_input: int = 0
    rows_output: int = 0
    execution_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    llm_tokens_used: int = 0
    llm_cost_usd: float = 0.0
    error_message: Optional[str] = None


@dataclass
class LLMUsageMetrics:
    """LLM usage and cost metrics"""
    provider: str
    model: str
    execution_id: str
    timestamp: datetime
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_seconds: float
    success: bool
    use_case: str  # enrichment, validation, transformation, etc.


# Prometheus metrics
pipeline_executions_total = Counter(
    'pipeline_executions_total',
    'Total number of pipeline executions',
    ['pipeline_name', 'status']
)

pipeline_execution_duration = Histogram(
    'pipeline_execution_duration_seconds',
    'Pipeline execution duration in seconds',
    ['pipeline_name']
)

step_execution_duration = Histogram(
    'step_execution_duration_seconds',
    'Step execution duration in seconds',
    ['step_name', 'step_type']
)

rows_processed_total = Counter(
    'rows_processed_total',
    'Total number of rows processed',
    ['pipeline_name', 'step_name']
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total LLM tokens used',
    ['provider', 'model', 'use_case']
)

llm_cost_total = Counter(
    'llm_cost_usd_total',
    'Total LLM cost in USD',
    ['provider', 'model']
)

active_pipelines = Gauge(
    'active_pipelines',
    'Number of currently running pipelines'
)

error_rate = Summary(
    'pipeline_error_rate',
    'Pipeline error rate',
    ['pipeline_name']
)


class MetricsCollector:
    """
    Centralized metrics collection and monitoring
    """
    
    def __init__(self):
        self.redis_client = None
        self.logger = structlog.get_logger()
        
        # Cost configuration (per 1M tokens)
        self.llm_costs = {
            "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gemini-pro": {"input": 0.5, "output": 1.5},
        }
        
        # Compute costs (per hour)
        self.compute_costs = {
            "small": 0.10,   # 2 vCPU, 4GB RAM
            "medium": 0.25,  # 4 vCPU, 8GB RAM
            "large": 0.50,   # 8 vCPU, 16GB RAM
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def track_pipeline_start(self, pipeline_metrics: PipelineMetrics):
        """Track pipeline execution start"""
        # Update Prometheus metrics
        active_pipelines.inc()
        
        # Store in Redis
        key = f"pipeline:{pipeline_metrics.execution_id}"
        await self.redis_client.hset(
            key,
            mapping={
                "pipeline_id": pipeline_metrics.pipeline_id,
                "pipeline_name": pipeline_metrics.pipeline_name,
                "start_time": pipeline_metrics.start_time.isoformat(),
                "status": "running",
                "total_steps": pipeline_metrics.total_steps
            }
        )
        
        # Set expiration (7 days)
        await self.redis_client.expire(key, 604800)
        
        # Add to active pipelines set
        await self.redis_client.sadd("active_pipelines", pipeline_metrics.execution_id)
        
        # Log
        self.logger.info(
            "pipeline_started",
            pipeline_id=pipeline_metrics.pipeline_id,
            pipeline_name=pipeline_metrics.pipeline_name,
            execution_id=pipeline_metrics.execution_id
        )
    
    async def track_pipeline_end(
        self,
        execution_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """Track pipeline execution completion"""
        # Get pipeline data
        key = f"pipeline:{execution_id}"
        pipeline_data = await self.redis_client.hgetall(key)
        
        if not pipeline_data:
            logger.warning(f"Pipeline metrics not found for execution {execution_id}")
            return
        
        # Calculate execution time
        start_time = datetime.fromisoformat(pipeline_data["start_time"])
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Update metrics
        pipeline_executions_total.labels(
            pipeline_name=pipeline_data["pipeline_name"],
            status=status
        ).inc()
        
        pipeline_execution_duration.labels(
            pipeline_name=pipeline_data["pipeline_name"]
        ).observe(execution_time)
        
        if status == "failed":
            error_rate.labels(
                pipeline_name=pipeline_data["pipeline_name"]
            ).observe(1)
        else:
            error_rate.labels(
                pipeline_name=pipeline_data["pipeline_name"]
            ).observe(0)
        
        # Update Redis
        await self.redis_client.hset(
            key,
            mapping={
                "end_time": end_time.isoformat(),
                "status": status,
                "execution_time_seconds": execution_time,
                "error_message": error_message or ""
            }
        )
        
        # Remove from active pipelines
        await self.redis_client.srem("active_pipelines", execution_id)
        active_pipelines.dec()
        
        # Calculate total cost
        total_cost = await self._calculate_pipeline_cost(execution_id)
        await self.redis_client.hset(key, "total_cost_usd", total_cost)
        
        # Log
        self.logger.info(
            "pipeline_completed",
            execution_id=execution_id,
            status=status,
            execution_time=execution_time,
            total_cost=total_cost
        )
    
    async def track_step_execution(self, step_metrics: StepMetrics):
        """Track individual step execution"""
        # Calculate execution time if completed
        if step_metrics.end_time:
            step_metrics.execution_time_seconds = (
                step_metrics.end_time - step_metrics.start_time
            ).total_seconds()
            
            # Update Prometheus metrics
            step_execution_duration.labels(
                step_name=step_metrics.step_name,
                step_type=step_metrics.step_type
            ).observe(step_metrics.execution_time_seconds)
            
            rows_processed_total.labels(
                pipeline_name="",  # TODO: Get from execution context
                step_name=step_metrics.step_name
            ).inc(step_metrics.rows_output)
        
        # Store in Redis
        key = f"step:{step_metrics.execution_id}:{step_metrics.step_name}"
        await self.redis_client.hset(
            key,
            mapping={
                "step_type": step_metrics.step_type,
                "start_time": step_metrics.start_time.isoformat(),
                "end_time": step_metrics.end_time.isoformat() if step_metrics.end_time else "",
                "status": step_metrics.status,
                "rows_input": step_metrics.rows_input,
                "rows_output": step_metrics.rows_output,
                "execution_time_seconds": step_metrics.execution_time_seconds,
                "memory_usage_mb": step_metrics.memory_usage_mb,
                "llm_tokens_used": step_metrics.llm_tokens_used,
                "llm_cost_usd": step_metrics.llm_cost_usd,
                "error_message": step_metrics.error_message or ""
            }
        )
        
        # Set expiration
        await self.redis_client.expire(key, 604800)
        
        # Update pipeline metrics
        pipeline_key = f"pipeline:{step_metrics.execution_id}"
        if step_metrics.status == "completed":
            await self.redis_client.hincrby(pipeline_key, "completed_steps", 1)
        elif step_metrics.status == "failed":
            await self.redis_client.hincrby(pipeline_key, "failed_steps", 1)
        
        # Update costs
        await self.redis_client.hincrbyfloat(
            pipeline_key,
            "llm_tokens_used",
            step_metrics.llm_tokens_used
        )
        await self.redis_client.hincrbyfloat(
            pipeline_key,
            "llm_cost_usd",
            step_metrics.llm_cost_usd
        )
    
    async def track_llm_usage(self, usage_metrics: LLMUsageMetrics):
        """Track LLM usage and costs"""
        # Update Prometheus metrics
        llm_tokens_total.labels(
            provider=usage_metrics.provider,
            model=usage_metrics.model,
            use_case=usage_metrics.use_case
        ).inc(usage_metrics.total_tokens)
        
        llm_cost_total.labels(
            provider=usage_metrics.provider,
            model=usage_metrics.model
        ).inc(usage_metrics.cost_usd)
        
        # Store detailed usage
        key = f"llm_usage:{usage_metrics.execution_id}:{usage_metrics.timestamp.timestamp()}"
        await self.redis_client.hset(
            key,
            mapping=asdict(usage_metrics)
        )
        
        # Set expiration
        await self.redis_client.expire(key, 2592000)  # 30 days
        
        # Update daily aggregates
        date_key = usage_metrics.timestamp.strftime("%Y-%m-%d")
        daily_key = f"llm_daily:{date_key}:{usage_metrics.provider}:{usage_metrics.model}"
        
        await self.redis_client.hincrby(daily_key, "total_tokens", usage_metrics.total_tokens)
        await self.redis_client.hincrbyfloat(daily_key, "total_cost", usage_metrics.cost_usd)
        await self.redis_client.hincrby(daily_key, "request_count", 1)
        
        # Set expiration on daily aggregate
        await self.redis_client.expire(daily_key, 7776000)  # 90 days
    
    def calculate_llm_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate LLM cost based on token usage"""
        model_key = model
        if provider == "openai" and "gpt-4" in model:
            model_key = "gpt-4-turbo"
        elif provider == "openai" and "gpt-3.5" in model:
            model_key = "gpt-3.5-turbo"
        elif provider == "google":
            model_key = "gemini-pro"
        
        costs = self.llm_costs.get(model_key, {"input": 1.0, "output": 1.0})
        
        input_cost = (prompt_tokens / 1_000_000) * costs["input"]
        output_cost = (completion_tokens / 1_000_000) * costs["output"]
        
        return round(input_cost + output_cost, 6)
    
    def calculate_compute_cost(
        self,
        execution_time_seconds: float,
        instance_type: str = "small"
    ) -> float:
        """Calculate compute cost based on execution time"""
        hourly_rate = self.compute_costs.get(instance_type, 0.10)
        hours = execution_time_seconds / 3600
        return round(hours * hourly_rate, 6)
    
    async def _calculate_pipeline_cost(self, execution_id: str) -> float:
        """Calculate total pipeline cost"""
        pipeline_key = f"pipeline:{execution_id}"
        pipeline_data = await self.redis_client.hgetall(pipeline_key)
        
        llm_cost = float(pipeline_data.get("llm_cost_usd", 0))
        execution_time = float(pipeline_data.get("execution_time_seconds", 0))
        
        # Estimate compute cost
        compute_cost = self.calculate_compute_cost(execution_time)
        
        # Storage cost (simplified - $0.02 per GB per month)
        rows_processed = int(pipeline_data.get("rows_processed", 0))
        estimated_gb = rows_processed * 0.0001  # Rough estimate
        storage_cost = estimated_gb * 0.02 / 30  # Daily rate
        
        total_cost = llm_cost + compute_cost + storage_cost
        
        return round(total_cost, 4)
    
    async def get_pipeline_metrics(
        self,
        execution_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get pipeline execution metrics"""
        metrics = []
        
        if execution_id:
            # Get specific execution
            key = f"pipeline:{execution_id}"
            data = await self.redis_client.hgetall(key)
            if data:
                metrics.append(data)
        else:
            # Get all executions matching criteria
            # Note: In production, use a proper time-series database
            pattern = "pipeline:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    data = await self.redis_client.hgetall(key)
                    
                    # Filter by pipeline_id if specified
                    if pipeline_id and data.get("pipeline_id") != pipeline_id:
                        continue
                    
                    # Filter by date range if specified
                    if start_date or end_date:
                        exec_time = datetime.fromisoformat(data.get("start_time", ""))
                        if start_date and exec_time < start_date:
                            continue
                        if end_date and exec_time > end_date:
                            continue
                    
                    metrics.append(data)
                
                if cursor == 0:
                    break
        
        return metrics
    
    async def get_cost_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day"  # day, week, month
    ) -> Dict[str, Any]:
        """Get cost summary for date range"""
        summary = {
            "total_cost": 0.0,
            "llm_cost": 0.0,
            "compute_cost": 0.0,
            "storage_cost": 0.0,
            "by_provider": {},
            "by_pipeline": {},
            "timeline": []
        }
        
        # Get pipeline metrics for date range
        metrics = await self.get_pipeline_metrics(
            start_date=start_date,
            end_date=end_date
        )
        
        # Aggregate costs
        for metric in metrics:
            total_cost = float(metric.get("total_cost_usd", 0))
            llm_cost = float(metric.get("llm_cost_usd", 0))
            execution_time = float(metric.get("execution_time_seconds", 0))
            
            compute_cost = self.calculate_compute_cost(execution_time)
            storage_cost = total_cost - llm_cost - compute_cost
            
            summary["total_cost"] += total_cost
            summary["llm_cost"] += llm_cost
            summary["compute_cost"] += compute_cost
            summary["storage_cost"] += storage_cost
            
            # By pipeline
            pipeline_name = metric.get("pipeline_name", "unknown")
            if pipeline_name not in summary["by_pipeline"]:
                summary["by_pipeline"][pipeline_name] = 0.0
            summary["by_pipeline"][pipeline_name] += total_cost
        
        # Get LLM daily aggregates
        current_date = start_date
        while current_date <= end_date:
            date_key = current_date.strftime("%Y-%m-%d")
            daily_cost = 0.0
            
            # Scan for daily LLM costs
            pattern = f"llm_daily:{date_key}:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    data = await self.redis_client.hgetall(key)
                    cost = float(data.get("total_cost", 0))
                    daily_cost += cost
                    
                    # By provider
                    parts = key.split(":")
                    if len(parts) >= 3:
                        provider = parts[2]
                        if provider not in summary["by_provider"]:
                            summary["by_provider"][provider] = 0.0
                        summary["by_provider"][provider] += cost
                
                if cursor == 0:
                    break
            
            summary["timeline"].append({
                "date": date_key,
                "cost": daily_cost
            })
            
            current_date += timedelta(days=1)
        
        return summary
    
    async def get_active_pipelines(self) -> List[Dict[str, Any]]:
        """Get currently active pipelines"""
        active_ids = await self.redis_client.smembers("active_pipelines")
        active_pipelines_data = []
        
        for execution_id in active_ids:
            key = f"pipeline:{execution_id}"
            data = await self.redis_client.hgetall(key)
            if data:
                # Calculate current duration
                start_time = datetime.fromisoformat(data["start_time"])
                current_duration = (datetime.now() - start_time).total_seconds()
                data["current_duration_seconds"] = current_duration
                
                active_pipelines_data.append(data)
        
        return active_pipelines_data
    
    async def get_performance_stats(
        self,
        pipeline_id: Optional[str] = None,
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Get performance statistics"""
        end_date = datetime.now()
        start_date = end_date - time_window
        
        metrics = await self.get_pipeline_metrics(
            pipeline_id=pipeline_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not metrics:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_rows_processed": 0.0,
                "avg_cost": 0.0
            }
        
        total_executions = len(metrics)
        successful = sum(1 for m in metrics if m.get("status") == "completed")
        
        execution_times = [
            float(m.get("execution_time_seconds", 0))
            for m in metrics
            if m.get("execution_time_seconds")
        ]
        
        rows_processed = [
            int(m.get("rows_processed", 0))
            for m in metrics
            if m.get("rows_processed")
        ]
        
        costs = [
            float(m.get("total_cost_usd", 0))
            for m in metrics
            if m.get("total_cost_usd")
        ]
        
        return {
            "total_executions": total_executions,
            "success_rate": (successful / total_executions) * 100 if total_executions > 0 else 0,
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "avg_rows_processed": sum(rows_processed) / len(rows_processed) if rows_processed else 0,
            "avg_cost": sum(costs) / len(costs) if costs else 0,
            "total_cost": sum(costs),
            "total_rows": sum(rows_processed)
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()