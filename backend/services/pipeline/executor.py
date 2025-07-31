import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import pandas as pd
from collections import defaultdict

from .base import (
    PipelineConfig, PipelineExecution, PipelineStatus, 
    StepConfig, StepResult, StepStatus
)
from .step_registry import StepRegistry
from .state_manager import PipelineStateManager
from .optimizer import PipelineOptimizer
from services.cache_service import CacheService
from core.logging import get_logger

logger = get_logger(__name__)


class PipelineExecutor:
    """Executes data processing pipelines with optimization and fault tolerance"""
    
    def __init__(
        self,
        state_manager: Optional[PipelineStateManager] = None,
        optimizer: Optional[PipelineOptimizer] = None,
        max_workers: int = 4
    ):
        self.state_manager = state_manager or PipelineStateManager()
        self.optimizer = optimizer or PipelineOptimizer()
        self.cache = CacheService()
        self.max_workers = max_workers
        self._running_executions: Dict[str, PipelineExecution] = {}
    
    async def execute(
        self,
        config: PipelineConfig,
        initial_data: Optional[Any] = None,
        resume_from: Optional[str] = None
    ) -> PipelineExecution:
        """Execute a pipeline with optional resume capability"""
        
        # Create or resume execution
        if resume_from:
            execution = await self._resume_execution(resume_from)
            if not execution:
                raise ValueError(f"Cannot resume execution {resume_from}")
        else:
            execution = await self._create_execution(config)
        
        # Store in running executions
        self._running_executions[execution.id] = execution
        
        try:
            # Optimize pipeline if enabled
            if config.auto_optimize and not execution.optimization_applied:
                await self._optimize_pipeline(execution)
            
            # Build execution graph
            graph = self._build_dependency_graph(execution.pipeline_config.steps)
            
            # Execute pipeline
            await self._execute_pipeline(execution, graph, initial_data)
            
            # Finalize execution
            execution.status = PipelineStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.total_execution_time = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            
            # Save final state
            await self.state_manager.save_state(execution)
            
            return execution
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            execution.status = PipelineStatus.FAILED
            execution.error = str(e)
            await self.state_manager.save_state(execution)
            raise
            
        finally:
            # Remove from running executions
            self._running_executions.pop(execution.id, None)
    
    async def _create_execution(self, config: PipelineConfig) -> PipelineExecution:
        """Create a new pipeline execution"""
        execution = PipelineExecution(
            pipeline_id=config.id,
            pipeline_config=config,
            total_steps=len(config.steps),
            started_at=datetime.utcnow()
        )
        
        execution.status = PipelineStatus.RUNNING
        
        # Save initial state
        await self.state_manager.save_state(execution)
        
        logger.info(f"Created pipeline execution {execution.id}")
        return execution
    
    async def _resume_execution(self, execution_id: str) -> Optional[PipelineExecution]:
        """Resume a paused or failed execution"""
        execution = await self.state_manager.load_state(execution_id)
        
        if not execution:
            return None
        
        # Reset status for resume
        if execution.status in [PipelineStatus.FAILED, PipelineStatus.PAUSED]:
            execution.status = PipelineStatus.RUNNING
            logger.info(f"Resuming pipeline execution {execution_id}")
        
        return execution
    
    async def _optimize_pipeline(self, execution: PipelineExecution):
        """Apply pipeline optimization"""
        try:
            execution.status = PipelineStatus.OPTIMIZING
            await self.state_manager.save_state(execution)
            
            # Optimize the pipeline configuration
            optimized_config = await self.optimizer.optimize(execution.pipeline_config)
            
            # Update execution with optimized config
            execution.pipeline_config = optimized_config
            execution.optimization_applied = True
            execution.optimization_details = self.optimizer.get_optimization_report()
            
            logger.info(f"Applied optimization to pipeline {execution.id}")
            
        except Exception as e:
            logger.warning(f"Pipeline optimization failed: {str(e)}")
            # Continue with original pipeline
        
        finally:
            execution.status = PipelineStatus.RUNNING
    
    def _build_dependency_graph(
        self, 
        steps: List[StepConfig]
    ) -> Dict[str, Set[str]]:
        """Build dependency graph from step configurations"""
        graph = defaultdict(set)
        step_names = {step.name for step in steps}
        
        for step in steps:
            # Validate dependencies
            for dep in step.depends_on:
                if dep not in step_names:
                    raise ValueError(
                        f"Step {step.name} depends on unknown step {dep}"
                    )
                graph[step.name].add(dep)
        
        # Check for cycles
        if self._has_cycle(graph):
            raise ValueError("Pipeline contains circular dependencies")
        
        return dict(graph)
    
    def _has_cycle(self, graph: Dict[str, Set[str]]) -> bool:
        """Check if dependency graph has cycles"""
        visited = set()
        rec_stack = set()
        
        def visit(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if visit(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        return any(visit(node) for node in graph if node not in visited)
    
    async def _execute_pipeline(
        self,
        execution: PipelineExecution,
        graph: Dict[str, Set[str]],
        initial_data: Optional[Any] = None
    ):
        """Execute pipeline steps respecting dependencies"""
        # Track step outputs
        step_outputs: Dict[str, Any] = {}
        
        # Get steps that can be resumed
        completed_steps = set(
            name for name, result in execution.step_results.items()
            if result.status == StepStatus.COMPLETED
        )
        
        # Load outputs of completed steps
        for step_name in completed_steps:
            output = await self.state_manager.load_step_output(
                execution.id, step_name
            )
            if output is not None:
                step_outputs[step_name] = output
        
        # Create step execution tasks
        steps_by_name = {step.name: step for step in execution.pipeline_config.steps}
        pending_steps = set(steps_by_name.keys()) - completed_steps
        
        # Execute steps in topological order with parallelism
        while pending_steps:
            # Find steps that can run now
            ready_steps = [
                step_name for step_name in pending_steps
                if all(dep in completed_steps for dep in graph.get(step_name, []))
            ]
            
            if not ready_steps:
                raise RuntimeError("No steps ready to execute - possible dependency issue")
            
            # Execute ready steps in parallel (up to max_workers)
            batch_size = min(len(ready_steps), self.max_workers)
            batch = ready_steps[:batch_size]
            
            # Create tasks for parallel execution
            tasks = []
            for step_name in batch:
                step_config = steps_by_name[step_name]
                
                # Determine input data
                if step_config.depends_on:
                    # Use output from dependency
                    # If multiple dependencies, pass as dict
                    if len(step_config.depends_on) == 1:
                        input_data = step_outputs.get(step_config.depends_on[0])
                    else:
                        input_data = {
                            dep: step_outputs.get(dep)
                            for dep in step_config.depends_on
                        }
                else:
                    # Use initial data for root steps
                    input_data = initial_data
                
                # Create execution task
                task = self._execute_step(
                    execution, step_config, input_data, step_outputs
                )
                tasks.append(task)
            
            # Wait for batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step_name, result in zip(batch, results):
                if isinstance(result, Exception):
                    raise result
                
                completed_steps.add(step_name)
                pending_steps.remove(step_name)
                
                # Update execution progress
                execution.current_step = step_name
                
                # Checkpoint if enabled
                if execution.pipeline_config.checkpoint_enabled:
                    if execution.completed_steps % execution.pipeline_config.checkpoint_interval == 0:
                        await self.state_manager.checkpoint(execution, result)
    
    async def _execute_step(
        self,
        execution: PipelineExecution,
        step_config: StepConfig,
        input_data: Any,
        step_outputs: Dict[str, Any]
    ) -> StepResult:
        """Execute a single pipeline step"""
        logger.info(f"Executing step {step_config.name}")
        
        # Check cache if enabled
        if step_config.cache_enabled:
            cached_result = await self._get_cached_result(step_config, input_data)
            if cached_result:
                logger.info(f"Using cached result for step {step_config.name}")
                execution.step_results[step_config.name] = cached_result
                step_outputs[step_config.name] = cached_result.output_data
                return cached_result
        
        # Create step instance
        step = StepRegistry.create_step(step_config)
        
        # Prepare context
        context = {
            "execution_id": execution.id,
            "pipeline_config": execution.pipeline_config,
            "step_outputs": step_outputs,
            "llm_budget_remaining": (
                execution.pipeline_config.max_llm_cost - execution.total_llm_cost
                if execution.pipeline_config.max_llm_cost else None
            )
        }
        
        # Execute with retries
        retry_count = 0
        last_error = None
        
        while retry_count <= step_config.retry_count:
            try:
                # Execute step
                result = await asyncio.wait_for(
                    step.execute(input_data, context),
                    timeout=step_config.timeout
                )
                
                # Update execution metrics
                if result.llm_tokens_used:
                    execution.total_llm_tokens += result.llm_tokens_used
                if result.llm_cost:
                    execution.total_llm_cost += result.llm_cost
                if result.row_count:
                    execution.total_rows_processed += result.row_count
                
                # Store result
                execution.step_results[step_config.name] = result
                step_outputs[step_config.name] = result.output_data
                
                # Cache result if enabled
                if step_config.cache_enabled and result.status == StepStatus.COMPLETED:
                    await self._cache_result(step_config, input_data, result)
                
                # Save step output if large
                if result.output_data is not None and hasattr(result.output_data, '__len__'):
                    if len(result.output_data) > 1000:
                        await self.state_manager.save_step_output(
                            execution.id, step_config.name, result.output_data
                        )
                
                return result
                
            except asyncio.TimeoutError:
                last_error = f"Step {step_config.name} timed out after {step_config.timeout}s"
                logger.error(last_error)
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Step {step_config.name} failed: {last_error}")
            
            retry_count += 1
            if retry_count <= step_config.retry_count:
                wait_time = execution.pipeline_config.retry_delay * retry_count
                logger.info(f"Retrying step {step_config.name} in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        result = StepResult(
            step_name=step_config.name,
            status=StepStatus.FAILED,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            error=last_error
        )
        
        execution.step_results[step_config.name] = result
        
        # Handle failure based on error policy
        if execution.pipeline_config.on_error == "fail":
            execution.status = PipelineStatus.FAILED
            execution.failed_step = step_config.name
            raise RuntimeError(f"Step {step_config.name} failed: {last_error}")
        
        return result
    
    async def _get_cached_result(
        self,
        step_config: StepConfig,
        input_data: Any
    ) -> Optional[StepResult]:
        """Get cached result for a step"""
        step = StepRegistry.create_step(step_config)
        cache_key = step.get_cache_key(input_data)
        
        cached = await self.cache.get(cache_key)
        if cached:
            return StepResult(**cached)
        
        return None
    
    async def _cache_result(
        self,
        step_config: StepConfig,
        input_data: Any,
        result: StepResult
    ):
        """Cache step result"""
        step = StepRegistry.create_step(step_config)
        cache_key = step.get_cache_key(input_data)
        
        # Don't cache the actual data, just the result metadata
        result_to_cache = result.dict()
        result_to_cache['output_data'] = None  # Clear large data
        
        await self.cache.set(
            cache_key,
            result_to_cache,
            ttl=step_config.cache_ttl
        )
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution"""
        execution = self._running_executions.get(execution_id)
        
        if not execution:
            return False
        
        execution.status = PipelineStatus.PAUSED
        await self.state_manager.save_state(execution)
        
        logger.info(f"Paused pipeline execution {execution_id}")
        return True
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        execution = self._running_executions.get(execution_id)
        
        if not execution:
            return False
        
        execution.status = PipelineStatus.CANCELLED
        await self.state_manager.save_state(execution)
        
        logger.info(f"Cancelled pipeline execution {execution_id}")
        return True
    
    def get_running_executions(self) -> List[str]:
        """Get list of currently running execution IDs"""
        return list(self._running_executions.keys())