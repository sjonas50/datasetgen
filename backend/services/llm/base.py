from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
from datetime import datetime
import json
from enum import Enum

from core.logging import get_logger

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    LLAMA = "llama"


class LLMResponse(BaseModel):
    """Standard response from LLM services"""
    content: str
    model: str
    usage: Dict[str, int]  # tokens used, cost, etc.
    metadata: Dict[str, Any] = {}
    timestamp: datetime = datetime.utcnow()
    cached: bool = False


class LLMConfig(BaseModel):
    """Configuration for LLM requests"""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    stream: bool = False
    system_prompt: Optional[str] = None
    response_format: Optional[str] = None  # "json" or "text"
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None
    
    # Claude-specific
    enable_extended_thinking: bool = False
    enable_vision: bool = False
    
    # Cost optimization
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour


class BaseLLMService(ABC):
    """Abstract base class for LLM services"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._usage_stats = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "request_count": 0
        }
    
    @abstractmethod
    async def complete(
        self, 
        prompt: str, 
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion for a prompt"""
        pass
    
    @abstractmethod
    async def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion with tool/function calling"""
        pass
    
    @abstractmethod
    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Analyze an image with prompt"""
        pass
    
    async def infer_schema(
        self, 
        data_sample: Union[Dict[str, Any], List[Dict[str, Any]]],
        config: Optional[LLMConfig] = None
    ) -> Dict[str, Any]:
        """Infer data schema from sample"""
        prompt = self._build_schema_inference_prompt(data_sample)
        
        if config is None:
            config = LLMConfig(
                temperature=0.3,
                response_format="json",
                enable_extended_thinking=True
            )
        
        response = await self.complete(prompt, config)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse schema response: {response.content}")
            return {}
    
    async def generate_transformation_code(
        self,
        source_schema: Dict[str, Any],
        target_schema: Dict[str, Any],
        language: str = "python",
        config: Optional[LLMConfig] = None
    ) -> str:
        """Generate code to transform data from source to target schema"""
        prompt = self._build_transformation_prompt(
            source_schema, target_schema, language
        )
        
        if config is None:
            config = LLMConfig(
                temperature=0.2,
                max_tokens=4000
            )
        
        response = await self.complete(prompt, config)
        return response.content
    
    async def assess_data_quality(
        self,
        data_stats: Dict[str, Any],
        config: Optional[LLMConfig] = None
    ) -> Dict[str, Any]:
        """Assess data quality and suggest improvements"""
        prompt = self._build_quality_assessment_prompt(data_stats)
        
        if config is None:
            config = LLMConfig(
                temperature=0.5,
                response_format="json"
            )
        
        response = await self.complete(prompt, config)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"error": "Failed to parse quality assessment"}
    
    def _build_schema_inference_prompt(self, data_sample: Any) -> str:
        """Build prompt for schema inference"""
        return f"""Analyze this data sample and infer a comprehensive schema.

Data Sample:
{json.dumps(data_sample, indent=2)[:2000]}  # Truncate if too long

Return a JSON object with:
1. "columns": List of column definitions with:
   - name: Column name
   - type: Data type (string, integer, float, boolean, datetime, object, array)
   - nullable: Whether nulls are allowed
   - description: Semantic meaning of the column
   - format: Specific format if applicable (e.g., "email", "phone", "date")
   - constraints: Any detected constraints (min, max, pattern, enum values)

2. "relationships": Detected relationships between columns
3. "quality_issues": Any data quality issues detected
4. "recommendations": Suggestions for data improvement

Focus on semantic understanding, not just basic type detection."""
    
    def _build_transformation_prompt(
        self, 
        source_schema: Dict[str, Any],
        target_schema: Dict[str, Any],
        language: str
    ) -> str:
        """Build prompt for transformation code generation"""
        return f"""Generate {language} code to transform data from the source schema to the target schema.

Source Schema:
{json.dumps(source_schema, indent=2)}

Target Schema:
{json.dumps(target_schema, indent=2)}

Requirements:
1. Handle data type conversions
2. Apply necessary validations
3. Handle missing or null values appropriately
4. Include error handling
5. Make the code efficient and readable
6. Add comments explaining complex transformations

Return only the code, no explanations."""
    
    def _build_quality_assessment_prompt(self, data_stats: Dict[str, Any]) -> str:
        """Build prompt for data quality assessment"""
        return f"""Assess the data quality based on these statistics:

{json.dumps(data_stats, indent=2)}

Return a JSON object with:
1. "overall_score": Quality score from 0-100
2. "issues": List of identified issues with severity (high/medium/low)
3. "recommendations": Specific actionable recommendations
4. "metrics": Detailed quality metrics per column
5. "data_profile": Summary of data characteristics

Consider completeness, consistency, accuracy, and validity."""
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self._usage_stats.copy()
    
    def _update_usage(self, tokens: int, cost: float):
        """Update usage statistics"""
        self._usage_stats["total_tokens"] += tokens
        self._usage_stats["total_cost"] += cost
        self._usage_stats["request_count"] += 1