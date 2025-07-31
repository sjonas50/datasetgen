from typing import Dict, Any, List, Optional, Tuple
import copy
from collections import defaultdict

from .base import PipelineConfig, StepConfig, StepType
from services.llm.llm_factory import LLMFactory
from core.logging import get_logger

logger = get_logger(__name__)


class PipelineOptimizer:
    """
    DocETL-inspired pipeline optimizer that rewrites pipelines for efficiency
    
    Key optimizations:
    1. Step reordering for parallelism
    2. Filter pushdown to reduce data volume early
    3. LLM call batching and deduplication
    4. Automatic caching insertion
    5. Cost-based routing to cheaper models
    """
    
    def __init__(self):
        self._optimization_report = {}
        self.llm_service = None
    
    async def optimize(self, config: PipelineConfig) -> PipelineConfig:
        """Optimize pipeline configuration"""
        logger.info(f"Optimizing pipeline {config.name}")
        
        # Reset report
        self._optimization_report = {
            "original_steps": len(config.steps),
            "optimizations_applied": [],
            "estimated_cost_reduction": 0.0,
            "estimated_time_reduction": 0.0
        }
        
        # Create a copy to modify
        optimized_config = copy.deepcopy(config)
        
        # Apply optimization strategies
        if config.optimization_level >= 1:
            # Basic optimizations
            optimized_config = await self._optimize_filter_pushdown(optimized_config)
            optimized_config = await self._optimize_step_ordering(optimized_config)
            optimized_config = await self._optimize_caching(optimized_config)
        
        if config.optimization_level >= 2:
            # Advanced optimizations
            optimized_config = await self._optimize_llm_usage(optimized_config)
            optimized_config = await self._optimize_parallel_execution(optimized_config)
            optimized_config = await self._optimize_data_sampling(optimized_config)
        
        # Update report
        self._optimization_report["optimized_steps"] = len(optimized_config.steps)
        
        logger.info(
            f"Optimization complete. Applied {len(self._optimization_report['optimizations_applied'])} optimizations"
        )
        
        return optimized_config
    
    async def _optimize_filter_pushdown(self, config: PipelineConfig) -> PipelineConfig:
        """Push filter operations earlier in the pipeline to reduce data volume"""
        filter_steps = []
        other_steps = []
        
        # Separate filter steps
        for step in config.steps:
            if step.type == StepType.FILTER or (
                step.type == StepType.TRANSFORM and 
                step.config.get("type") == "filter"
            ):
                filter_steps.append(step)
            else:
                other_steps.append(step)
        
        if not filter_steps:
            return config
        
        # Analyze which filters can be pushed down
        pushable_filters = []
        for filter_step in filter_steps:
            # Check if filter can be pushed to the beginning
            if not filter_step.depends_on or all(
                dep in [s.name for s in pushable_filters]
                for dep in filter_step.depends_on
            ):
                pushable_filters.append(filter_step)
        
        if pushable_filters:
            # Reorder: pushable filters first, then others
            new_steps = pushable_filters + [
                s for s in other_steps if s not in pushable_filters
            ] + [
                s for s in filter_steps if s not in pushable_filters
            ]
            
            config.steps = new_steps
            
            self._optimization_report["optimizations_applied"].append({
                "type": "filter_pushdown",
                "description": f"Pushed {len(pushable_filters)} filters earlier",
                "estimated_benefit": "20-50% data reduction"
            })
        
        return config
    
    async def _optimize_step_ordering(self, config: PipelineConfig) -> PipelineConfig:
        """Reorder steps to maximize parallelism"""
        # Build dependency graph
        graph = defaultdict(set)
        reverse_graph = defaultdict(set)
        
        for step in config.steps:
            for dep in step.depends_on:
                graph[step.name].add(dep)
                reverse_graph[dep].add(step.name)
        
        # Topological sort with level assignment
        levels = self._assign_levels(config.steps, graph, reverse_graph)
        
        # Sort steps by level to maximize parallelism
        config.steps.sort(key=lambda s: (levels.get(s.name, 0), s.name))
        
        # Count potential parallel executions
        level_counts = defaultdict(int)
        for step_name, level in levels.items():
            level_counts[level] += 1
        
        max_parallel = max(level_counts.values()) if level_counts else 1
        
        if max_parallel > 1:
            self._optimization_report["optimizations_applied"].append({
                "type": "step_reordering",
                "description": f"Reordered for up to {max_parallel}x parallelism",
                "estimated_benefit": f"{(max_parallel - 1) * 20}% time reduction"
            })
        
        return config
    
    async def _optimize_caching(self, config: PipelineConfig) -> PipelineConfig:
        """Add caching to expensive operations"""
        cache_candidates = []
        
        for step in config.steps:
            # Identify expensive steps
            if step.type in [StepType.LLM_PROCESS, StepType.CONNECTOR]:
                # Enable caching if not already
                if not step.cache_enabled:
                    step.cache_enabled = True
                    step.cache_ttl = 7200  # 2 hours for expensive ops
                    cache_candidates.append(step.name)
            
            # Also cache heavy transforms
            elif step.type == StepType.TRANSFORM:
                transform_type = step.config.get("type")
                if transform_type in ["aggregate", "join", "custom"]:
                    if not step.cache_enabled:
                        step.cache_enabled = True
                        step.cache_ttl = 3600  # 1 hour
                        cache_candidates.append(step.name)
        
        if cache_candidates:
            self._optimization_report["optimizations_applied"].append({
                "type": "caching",
                "description": f"Added caching to {len(cache_candidates)} expensive steps",
                "estimated_benefit": "50-90% cost reduction on re-runs"
            })
        
        return config
    
    async def _optimize_llm_usage(self, config: PipelineConfig) -> PipelineConfig:
        """Optimize LLM usage for cost and efficiency"""
        llm_steps = [s for s in config.steps if s.type == StepType.LLM_PROCESS]
        
        if not llm_steps:
            return config
        
        optimizations_made = []
        
        for step in llm_steps:
            process_type = step.config.get("type", "")
            
            # Route simple tasks to cheaper models
            if process_type in ["validate", "filter"]:
                if not step.llm_model or "gpt-4" in step.llm_model:
                    step.llm_provider = "gemini"
                    step.llm_model = "gemini-2.5-flash"
                    optimizations_made.append(f"Routed {step.name} to cheaper model")
            
            # Enable batching for similar operations
            if process_type == "enrich":
                step.config["batch_size"] = 10
                step.config["batch_enabled"] = True
                optimizations_made.append(f"Enabled batching for {step.name}")
        
        # Look for consecutive LLM steps that can be combined
        i = 0
        while i < len(config.steps) - 1:
            current = config.steps[i]
            next_step = config.steps[i + 1]
            
            if (current.type == StepType.LLM_PROCESS and 
                next_step.type == StepType.LLM_PROCESS and
                next_step.depends_on == [current.name]):
                
                # Combine into single LLM call if possible
                if self._can_combine_llm_steps(current, next_step):
                    combined = self._combine_llm_steps(current, next_step)
                    config.steps[i] = combined
                    config.steps.pop(i + 1)
                    optimizations_made.append(f"Combined {current.name} and {next_step.name}")
                    continue
            
            i += 1
        
        if optimizations_made:
            self._optimization_report["optimizations_applied"].append({
                "type": "llm_optimization",
                "description": f"Applied {len(optimizations_made)} LLM optimizations",
                "estimated_benefit": "30-60% cost reduction",
                "details": optimizations_made
            })
        
        return config
    
    async def _optimize_parallel_execution(self, config: PipelineConfig) -> PipelineConfig:
        """Identify and mark steps for parallel execution"""
        # Increase max parallel steps if beneficial
        independent_groups = self._find_independent_step_groups(config.steps)
        
        if independent_groups:
            max_group_size = max(len(group) for group in independent_groups)
            if max_group_size > config.max_parallel_steps:
                config.max_parallel_steps = min(max_group_size, 8)  # Cap at 8
                
                self._optimization_report["optimizations_applied"].append({
                    "type": "parallel_execution",
                    "description": f"Increased parallelism to {config.max_parallel_steps}",
                    "estimated_benefit": f"{(config.max_parallel_steps - 4) * 15}% time reduction"
                })
        
        return config
    
    async def _optimize_data_sampling(self, config: PipelineConfig) -> PipelineConfig:
        """Add sampling for development/testing runs"""
        # This is more relevant for development mode
        if config.tags and "development" in config.tags:
            # Add sampling step after connectors
            for i, step in enumerate(config.steps):
                if step.type == StepType.CONNECTOR:
                    # Check if next step is not already a sample
                    if (i + 1 < len(config.steps) and 
                        config.steps[i + 1].config.get("type") != "sample"):
                        
                        sample_step = StepConfig(
                            name=f"{step.name}_sample",
                            type=StepType.TRANSFORM,
                            config={
                                "type": "sample",
                                "size": 1000,
                                "method": "random"
                            },
                            depends_on=[step.name]
                        )
                        
                        # Insert sample step
                        config.steps.insert(i + 1, sample_step)
                        
                        # Update dependencies of subsequent steps
                        for j in range(i + 2, len(config.steps)):
                            if step.name in config.steps[j].depends_on:
                                config.steps[j].depends_on = [
                                    sample_step.name if dep == step.name else dep
                                    for dep in config.steps[j].depends_on
                                ]
            
            self._optimization_report["optimizations_applied"].append({
                "type": "data_sampling",
                "description": "Added sampling for development mode",
                "estimated_benefit": "90% reduction in processing time"
            })
        
        return config
    
    def _assign_levels(
        self,
        steps: List[StepConfig],
        graph: Dict[str, set],
        reverse_graph: Dict[str, set]
    ) -> Dict[str, int]:
        """Assign levels to steps for parallel execution"""
        levels = {}
        visited = set()
        
        def assign_level(step_name: str) -> int:
            if step_name in levels:
                return levels[step_name]
            
            # Find max level of dependencies
            deps = graph.get(step_name, set())
            if not deps:
                level = 0
            else:
                level = max(assign_level(dep) for dep in deps) + 1
            
            levels[step_name] = level
            return level
        
        for step in steps:
            assign_level(step.name)
        
        return levels
    
    def _can_combine_llm_steps(self, step1: StepConfig, step2: StepConfig) -> bool:
        """Check if two LLM steps can be combined"""
        # Same provider and model
        if step1.llm_provider != step2.llm_provider:
            return False
        if step1.llm_model != step2.llm_model:
            return False
        
        # Compatible operation types
        type1 = step1.config.get("type", "")
        type2 = step2.config.get("type", "")
        
        compatible_types = {
            ("enrich", "validate"),
            ("validate", "enrich"),
            ("enrich", "enrich")
        }
        
        return (type1, type2) in compatible_types
    
    def _combine_llm_steps(self, step1: StepConfig, step2: StepConfig) -> StepConfig:
        """Combine two LLM steps into one"""
        combined = StepConfig(
            name=f"{step1.name}_{step2.name}",
            type=StepType.LLM_PROCESS,
            config={
                "type": "combined",
                "operations": [
                    {"name": step1.name, "config": step1.config},
                    {"name": step2.name, "config": step2.config}
                ]
            },
            depends_on=step1.depends_on,
            llm_provider=step1.llm_provider,
            llm_model=step1.llm_model
        )
        
        return combined
    
    def _find_independent_step_groups(
        self,
        steps: List[StepConfig]
    ) -> List[List[str]]:
        """Find groups of steps that can run in parallel"""
        # Build dependency graph
        graph = defaultdict(set)
        for step in steps:
            for dep in step.depends_on:
                graph[step.name].add(dep)
        
        # Find independent groups using graph coloring
        groups = []
        assigned = set()
        
        for step in steps:
            if step.name in assigned:
                continue
            
            # Find all steps that can run with this one
            group = [step.name]
            assigned.add(step.name)
            
            for other in steps:
                if other.name in assigned:
                    continue
                
                # Check if independent
                if (other.name not in graph[step.name] and 
                    step.name not in graph[other.name]):
                    
                    # Check against all in group
                    independent = True
                    for member in group:
                        if (other.name in graph[member] or 
                            member in graph[other.name]):
                            independent = False
                            break
                    
                    if independent:
                        group.append(other.name)
                        assigned.add(other.name)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed optimization report"""
        return self._optimization_report
    
    async def suggest_optimizations(
        self,
        config: PipelineConfig,
        execution_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Suggest optimizations based on pipeline and execution history"""
        suggestions = []
        
        # Analyze pipeline structure
        step_types = defaultdict(int)
        for step in config.steps:
            step_types[step.type] += 1
        
        # Suggest based on patterns
        if step_types[StepType.LLM_PROCESS] > 3:
            suggestions.append({
                "type": "reduce_llm_calls",
                "priority": "high",
                "description": "Consider combining multiple LLM steps",
                "potential_savings": "40-60% cost reduction"
            })
        
        if not any(s.cache_enabled for s in config.steps):
            suggestions.append({
                "type": "enable_caching",
                "priority": "medium",
                "description": "Enable caching for expensive operations",
                "potential_savings": "50-90% on repeated runs"
            })
        
        # Analyze execution history if provided
        if execution_history:
            avg_time = sum(e.get("execution_time", 0) for e in execution_history) / len(execution_history)
            if avg_time > 300:  # 5 minutes
                suggestions.append({
                    "type": "add_sampling",
                    "priority": "medium",
                    "description": "Consider sampling for development iterations",
                    "potential_savings": "90% time reduction in development"
                })
        
        return suggestions