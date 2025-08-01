from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid

from core.logging import get_logger

logger = get_logger(__name__)


class StepType(str, Enum):
    """Types of pipeline steps"""
    CONNECTOR = "connector"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    LLM_PROCESS = "llm_process"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    OUTPUT = "output"
    CUSTOM = "custom"


class StepStatus(str, Enum):
    """Step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class PipelineStatus(str, Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    OPTIMIZING = "optimizing"


class StepConfig(BaseModel):
    """Configuration for a pipeline step"""
    name: str
    type: StepType
    config: Dict[str, Any] = {}
    
    # Dependencies
    depends_on: List[str] = []  # Step names this depends on
    
    # Execution options
    retry_count: int = 3
    timeout: int = 3600  # seconds
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    # Resource requirements
    memory_mb: int = 512
    cpu_cores: float = 1.0
    gpu_required: bool = False
    
    # LLM options
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_config: Dict[str, Any] = {}


class StepResult(BaseModel):
    """Result from step execution"""
    step_name: str
    status: StepStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Data
    output_data: Optional[Any] = None
    output_schema: Optional[Dict[str, Any]] = None
    row_count: Optional[int] = None
    
    # Metrics
    execution_time: Optional[float] = None
    memory_used_mb: Optional[float] = None
    
    # LLM usage
    llm_tokens_used: Optional[int] = None
    llm_cost: Optional[float] = None
    
    # Errors
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Warnings and logs
    warnings: List[str] = []
    logs: List[str] = []
    
    class Config:
        arbitrary_types_allowed = True


class PipelineConfig(BaseModel):
    """Complete pipeline configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    version: str = "1.0"
    
    # Steps
    steps: List[StepConfig]
    
    # Global options
    max_parallel_steps: int = 4
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 5  # steps
    
    # Error handling
    on_error: str = "fail"  # fail, continue, retry
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    
    # Optimization
    auto_optimize: bool = True
    optimization_level: int = 1  # 0=none, 1=basic, 2=aggressive
    
    # Cost controls
    max_llm_cost: Optional[float] = None
    prefer_cached: bool = True
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = []


class PipelineExecution(BaseModel):
    """Pipeline execution instance"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id: str
    pipeline_config: PipelineConfig
    
    # Status
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress
    total_steps: int
    completed_steps: int = 0
    current_step: Optional[str] = None
    
    # Results
    step_results: Dict[str, StepResult] = {}
    final_output: Optional[Any] = None
    
    # Metrics
    total_execution_time: Optional[float] = None
    total_llm_tokens: int = 0
    total_llm_cost: float = 0.0
    total_rows_processed: int = 0
    
    # Optimization
    optimization_applied: bool = False
    optimization_details: Optional[Dict[str, Any]] = None
    
    # Errors
    error: Optional[str] = None
    failed_step: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class BasePipelineStep(ABC):
    """Abstract base class for pipeline steps"""
    
    def __init__(self, name: str, config: StepConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"{__name__}.{name}")
    
    @abstractmethod
    async def execute(
        self, 
        input_data: Any,
        context: Dict[str, Any]
    ) -> StepResult:
        """Execute the step"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate step configuration"""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_size: int) -> Dict[str, float]:
        """Estimate execution cost"""
        pass
    
    async def pre_execute(self, context: Dict[str, Any]):
        """Hook called before execution"""
        pass
    
    async def post_execute(self, result: StepResult, context: Dict[str, Any]):
        """Hook called after execution"""
        pass
    
    def get_cache_key(self, input_data: Any) -> str:
        """Generate cache key for this step"""
        import hashlib
        import json
        
        key_data = {
            "step_name": self.name,
            "step_type": self.config.type,
            "config": self.config.config,
            "input_hash": hashlib.md5(
                str(input_data).encode()
            ).hexdigest() if input_data is not None else "none"
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"pipeline_step_{hashlib.md5(key_str.encode()).hexdigest()}"


class StepInput(BaseModel):
    """Input for a pipeline step"""
    data: Any
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class StepOutput(BaseModel):
    """Output from a pipeline step"""
    data: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class PipelineStep(BasePipelineStep):
    """Concrete implementation of pipeline step for compatibility"""
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> StepResult:
        """Execute the step"""
        raise NotImplementedError("Subclass must implement execute method")
    
    def validate_config(self) -> bool:
        """Validate step configuration"""
        return True
    
    async def process(self, input_data: StepInput) -> StepOutput:
        """Process method for new-style steps"""
        raise NotImplementedError("Subclass must implement process method")