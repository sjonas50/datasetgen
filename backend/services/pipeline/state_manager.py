import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
import aiofiles
from datetime import datetime

from .base import PipelineExecution, StepResult
from services.cache_service import CacheService
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class PipelineStateManager:
    """Manages pipeline execution state for checkpointing and recovery"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("/tmp/pipeline_states")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.cache = CacheService()
    
    async def save_state(self, execution: PipelineExecution):
        """Save pipeline execution state"""
        try:
            # Save to cache for quick access
            cache_key = f"pipeline_state_{execution.id}"
            await self.cache.set(
                cache_key,
                execution.dict(),
                ttl=86400  # 24 hours
            )
            
            # Save to disk for persistence
            file_path = self._get_state_file_path(execution.id)
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(
                    execution.dict(),
                    default=str,  # Handle datetime serialization
                    indent=2
                ))
            
            logger.info(f"Saved state for pipeline execution {execution.id}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline state: {str(e)}")
            raise
    
    async def load_state(self, execution_id: str) -> Optional[PipelineExecution]:
        """Load pipeline execution state"""
        try:
            # Try cache first
            cache_key = f"pipeline_state_{execution_id}"
            cached_state = await self.cache.get(cache_key)
            
            if cached_state:
                return PipelineExecution(**cached_state)
            
            # Load from disk
            file_path = self._get_state_file_path(execution_id)
            
            if not file_path.exists():
                logger.warning(f"No state found for execution {execution_id}")
                return None
            
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                state_dict = json.loads(content)
            
            # Restore to cache
            await self.cache.set(cache_key, state_dict, ttl=86400)
            
            return PipelineExecution(**state_dict)
            
        except Exception as e:
            logger.error(f"Failed to load pipeline state: {str(e)}")
            return None
    
    async def save_step_output(
        self, 
        execution_id: str,
        step_name: str,
        output_data: Any
    ):
        """Save step output data separately (for large datasets)"""
        try:
            # For DataFrames and large objects, save as pickle
            file_path = self._get_step_output_path(execution_id, step_name)
            
            if hasattr(output_data, 'to_pickle'):
                # Save DataFrame
                output_data.to_pickle(file_path)
            else:
                # Save generic object
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(pickle.dumps(output_data))
            
            # Save reference in cache
            cache_key = f"step_output_{execution_id}_{step_name}"
            await self.cache.set(
                cache_key,
                {"file_path": str(file_path), "saved_at": datetime.utcnow().isoformat()},
                ttl=86400
            )
            
        except Exception as e:
            logger.error(f"Failed to save step output: {str(e)}")
            raise
    
    async def load_step_output(
        self,
        execution_id: str,
        step_name: str
    ) -> Optional[Any]:
        """Load step output data"""
        try:
            file_path = self._get_step_output_path(execution_id, step_name)
            
            if not file_path.exists():
                return None
            
            # Check file extension to determine how to load
            if file_path.suffix == '.pkl':
                import pandas as pd
                try:
                    # Try loading as DataFrame first
                    return pd.read_pickle(file_path)
                except:
                    # Fall back to generic pickle
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            
        except Exception as e:
            logger.error(f"Failed to load step output: {str(e)}")
            return None
    
    async def checkpoint(
        self,
        execution: PipelineExecution,
        step_result: StepResult
    ):
        """Create a checkpoint after step completion"""
        try:
            # Update execution with step result
            execution.step_results[step_result.step_name] = step_result
            execution.completed_steps += 1
            
            # Save state
            await self.save_state(execution)
            
            # Save step output if large
            if step_result.output_data is not None:
                if hasattr(step_result.output_data, '__len__'):
                    # Save separately if large
                    if len(step_result.output_data) > 1000:
                        await self.save_step_output(
                            execution.id,
                            step_result.step_name,
                            step_result.output_data
                        )
                        # Clear from memory in step result
                        step_result.output_data = None
            
            logger.info(f"Checkpointed pipeline {execution.id} after step {step_result.step_name}")
            
        except Exception as e:
            logger.error(f"Failed to checkpoint: {str(e)}")
            # Don't fail the pipeline due to checkpoint failure
    
    async def get_last_completed_step(
        self,
        execution_id: str
    ) -> Optional[str]:
        """Get the name of the last successfully completed step"""
        execution = await self.load_state(execution_id)
        
        if not execution:
            return None
        
        # Find last completed step in order
        completed_steps = [
            (name, result.completed_at)
            for name, result in execution.step_results.items()
            if result.status == "completed" and result.completed_at
        ]
        
        if not completed_steps:
            return None
        
        # Sort by completion time and return the latest
        completed_steps.sort(key=lambda x: x[1])
        return completed_steps[-1][0]
    
    async def cleanup_execution(self, execution_id: str):
        """Clean up state files for an execution"""
        try:
            # Remove from cache
            cache_key = f"pipeline_state_{execution_id}"
            await self.cache.delete(cache_key)
            
            # Remove state file
            state_file = self._get_state_file_path(execution_id)
            if state_file.exists():
                state_file.unlink()
            
            # Remove step outputs
            output_dir = self.storage_path / execution_id
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
            
            logger.info(f"Cleaned up state for execution {execution_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup execution state: {str(e)}")
    
    def _get_state_file_path(self, execution_id: str) -> Path:
        """Get path for state file"""
        return self.storage_path / f"{execution_id}_state.json"
    
    def _get_step_output_path(self, execution_id: str, step_name: str) -> Path:
        """Get path for step output file"""
        output_dir = self.storage_path / execution_id
        output_dir.mkdir(exist_ok=True)
        return output_dir / f"{step_name}_output.pkl"
    
    async def list_executions(self) -> List[Dict[str, Any]]:
        """List all saved executions"""
        executions = []
        
        for state_file in self.storage_path.glob("*_state.json"):
            try:
                async with aiofiles.open(state_file, 'r') as f:
                    content = await f.read()
                    state = json.loads(content)
                
                executions.append({
                    "id": state["id"],
                    "pipeline_name": state["pipeline_config"]["name"],
                    "status": state["status"],
                    "started_at": state.get("started_at"),
                    "completed_at": state.get("completed_at"),
                    "progress": f"{state['completed_steps']}/{state['total_steps']}"
                })
            except Exception as e:
                logger.warning(f"Failed to read state file {state_file}: {str(e)}")
        
        return executions