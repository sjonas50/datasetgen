from fastapi import APIRouter, Query, HTTPException
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from services.monitoring.metrics_collector import metrics_collector
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/metrics/pipelines")
async def get_pipeline_metrics(
    execution_id: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """
    Get pipeline execution metrics
    
    Query parameters:
    - execution_id: Get metrics for specific execution
    - pipeline_id: Filter by pipeline ID
    - start_date: Filter by start date
    - end_date: Filter by end date
    - limit: Maximum number of results
    """
    try:
        metrics = await metrics_collector.get_pipeline_metrics(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Apply limit
        metrics = metrics[:limit]
        
        return {
            "total": len(metrics),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Failed to get pipeline metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get("/metrics/active-pipelines")
async def get_active_pipelines() -> Dict[str, Any]:
    """
    Get currently active pipeline executions
    """
    try:
        active = await metrics_collector.get_active_pipelines()
        
        return {
            "count": len(active),
            "pipelines": active
        }
    except Exception as e:
        logger.error(f"Failed to get active pipelines: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active pipelines")


@router.get("/metrics/costs")
async def get_cost_summary(
    start_date: datetime = Query(..., description="Start date for cost analysis"),
    end_date: datetime = Query(..., description="End date for cost analysis"),
    group_by: str = Query("day", regex="^(day|week|month)$")
) -> Dict[str, Any]:
    """
    Get cost summary for date range
    
    Returns breakdown by:
    - Total costs
    - LLM costs
    - Compute costs
    - Storage costs
    - Costs by provider
    - Costs by pipeline
    - Daily timeline
    """
    try:
        summary = await metrics_collector.get_cost_summary(
            start_date=start_date,
            end_date=end_date,
            group_by=group_by
        )
        
        return summary
    except Exception as e:
        logger.error(f"Failed to get cost summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cost summary")


@router.get("/metrics/performance")
async def get_performance_stats(
    pipeline_id: Optional[str] = None,
    days: int = Query(7, ge=1, le=90, description="Time window in days")
) -> Dict[str, Any]:
    """
    Get performance statistics
    
    Returns:
    - Total executions
    - Success rate
    - Average execution time
    - Average rows processed
    - Average cost per execution
    """
    try:
        stats = await metrics_collector.get_performance_stats(
            pipeline_id=pipeline_id,
            time_window=timedelta(days=days)
        )
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get performance stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance statistics")


@router.get("/metrics/llm-usage")
async def get_llm_usage(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get LLM usage statistics
    """
    # This would query the LLM usage data
    # For now, return aggregated data from cost summary
    try:
        cost_summary = await metrics_collector.get_cost_summary(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "total_llm_cost": cost_summary.get("llm_cost", 0),
            "by_provider": cost_summary.get("by_provider", {}),
            "timeline": cost_summary.get("timeline", [])
        }
    except Exception as e:
        logger.error(f"Failed to get LLM usage: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve LLM usage")


@router.get("/metrics/dashboard")
async def get_dashboard_data(
    days: int = Query(7, ge=1, le=30)
) -> Dict[str, Any]:
    """
    Get comprehensive dashboard data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get various metrics in parallel
        import asyncio
        
        active_pipelines_task = metrics_collector.get_active_pipelines()
        performance_task = metrics_collector.get_performance_stats(time_window=timedelta(days=days))
        cost_task = metrics_collector.get_cost_summary(start_date=start_date, end_date=end_date)
        recent_executions_task = metrics_collector.get_pipeline_metrics(
            start_date=start_date,
            end_date=end_date
        )
        
        active_pipelines, performance, costs, recent_executions = await asyncio.gather(
            active_pipelines_task,
            performance_task,
            cost_task,
            recent_executions_task
        )
        
        # Calculate additional metrics
        failed_executions = [
            e for e in recent_executions
            if e.get("status") == "failed"
        ][:10]  # Last 10 failures
        
        # Pipeline execution trends
        daily_executions = {}
        for execution in recent_executions:
            date = datetime.fromisoformat(execution["start_time"]).date().isoformat()
            if date not in daily_executions:
                daily_executions[date] = {"total": 0, "successful": 0, "failed": 0}
            
            daily_executions[date]["total"] += 1
            if execution.get("status") == "completed":
                daily_executions[date]["successful"] += 1
            elif execution.get("status") == "failed":
                daily_executions[date]["failed"] += 1
        
        execution_trend = [
            {"date": date, **stats}
            for date, stats in sorted(daily_executions.items())
        ]
        
        return {
            "summary": {
                "active_pipelines": len(active_pipelines),
                "total_executions_period": performance["total_executions"],
                "success_rate": performance["success_rate"],
                "total_cost_period": costs["total_cost"],
                "avg_execution_time": performance["avg_execution_time"],
                "total_rows_processed": performance["total_rows"]
            },
            "active_pipelines": active_pipelines,
            "recent_failures": failed_executions,
            "cost_breakdown": {
                "llm": costs["llm_cost"],
                "compute": costs["compute_cost"],
                "storage": costs["storage_cost"],
                "by_pipeline": costs["by_pipeline"],
                "by_provider": costs["by_provider"]
            },
            "execution_trend": execution_trend,
            "cost_trend": costs["timeline"],
            "period_days": days
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


@router.post("/metrics/alert-test")
async def test_alert(
    alert_type: str = Query(..., regex="^(cost_threshold|error_rate|execution_time)$"),
    threshold: float = Query(...)
) -> Dict[str, Any]:
    """
    Test monitoring alerts
    """
    # This would trigger test alerts based on type and threshold
    # For now, just return success
    return {
        "status": "success",
        "message": f"Test alert for {alert_type} with threshold {threshold} would be triggered"
    }