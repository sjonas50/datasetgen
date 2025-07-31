import anthropic
from anthropic import AsyncAnthropic
from typing import Dict, Any, Optional, List, Union
import base64
from pathlib import Path
import json
import asyncio
from datetime import datetime

from .base import BaseLLMService, LLMResponse, LLMConfig
from core.config import settings
from core.logging import get_logger
from services.cache_service import CacheService

logger = get_logger(__name__)


class ClaudeService(BaseLLMService):
    """Claude Sonnet 4 implementation with extended thinking and vision capabilities"""
    
    # Model configurations
    MODELS = {
        "sonnet-4": "claude-sonnet-4-20250514",
        "opus-4": "claude-opus-4-20250514",
        "haiku-3": "claude-3-haiku-20240307"
    }
    
    # Pricing per million tokens
    PRICING = {
        "claude-sonnet-4-20250514": {
            "input": 3.0,    # $3 per million
            "output": 15.0,  # $15 per million
            "cache_write": 3.75,
            "cache_read": 0.30
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "sonnet-4"):
        api_key = api_key or settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        model_name = self.MODELS.get(model, model)
        super().__init__(api_key, model_name)
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.cache_service = CacheService()
        
        # Extended thinking configuration
        self.extended_thinking_enabled = True
        self.max_thinking_time = 30  # seconds
    
    async def complete(
        self, 
        prompt: str, 
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion with optional extended thinking"""
        config = config or LLMConfig()
        
        # Check cache if enabled
        if config.use_cache:
            cache_key = self._generate_cache_key(prompt, config)
            cached_response = await self.cache_service.get(cache_key)
            if cached_response:
                logger.info("Returning cached response")
                return LLMResponse(**cached_response, cached=True)
        
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt, config.system_prompt)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": config.max_tokens or 4096,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
            
            # Add extended thinking if enabled
            if config.enable_extended_thinking:
                request_params["anthropic_beta"] = "extended-thinking-2025-01-24"
                request_params["extended_thinking"] = {
                    "enabled": True,
                    "max_thinking_time": self.max_thinking_time
                }
            
            # Add tools if provided
            if config.tools:
                request_params["tools"] = config.tools
                if config.tool_choice:
                    request_params["tool_choice"] = config.tool_choice
            
            # Make API call
            response = await self.client.messages.create(**request_params)
            
            # Extract content
            content = self._extract_content(response)
            
            # Calculate usage and cost
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "cost": self._calculate_cost(response.usage)
            }
            
            # Update usage stats
            self._update_usage(usage["total_tokens"], usage["cost"])
            
            # Create response
            llm_response = LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata={
                    "thinking_time": getattr(response, "thinking_time", None),
                    "stop_reason": response.stop_reason
                }
            )
            
            # Cache response if enabled
            if config.use_cache:
                await self.cache_service.set(
                    cache_key, 
                    llm_response.dict(), 
                    ttl=config.cache_ttl
                )
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            raise
    
    async def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion with tool/function calling"""
        config = config or LLMConfig()
        config.tools = tools
        
        # Enable interleaved thinking for better tool use
        config.enable_extended_thinking = True
        
        return await self.complete(prompt, config)
    
    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Analyze image with Claude's vision capabilities"""
        config = config or LLMConfig()
        
        # Read and encode image
        image_data = self._encode_image(image_path)
        
        # Prepare message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": self._get_media_type(image_path),
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=config.max_tokens or 4096,
                temperature=config.temperature
            )
            
            content = self._extract_content(response)
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "cost": self._calculate_cost(response.usage)
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata={"vision_analysis": True}
            )
            
        except Exception as e:
            logger.error(f"Vision analysis error: {str(e)}")
            raise
    
    async def analyze_document_with_vision(
        self,
        document_path: str,
        analysis_type: str = "extract_tables"
    ) -> Dict[str, Any]:
        """Analyze documents (PDF, images) for data extraction"""
        prompts = {
            "extract_tables": """Analyze this document and extract all tables.
For each table:
1. Identify column headers
2. Extract all row data
3. Note any data quality issues
4. Suggest appropriate data types

Return the data in JSON format with proper structure.""",
            
            "extract_structure": """Analyze this document's structure and content.
Identify:
1. Main sections and hierarchy
2. Key data elements and their relationships
3. Any forms or structured data
4. Metadata (dates, authors, references)

Return a comprehensive structural analysis in JSON.""",
            
            "extract_charts": """Analyze any charts, graphs, or visualizations in this document.
For each visualization:
1. Identify the type (bar, line, pie, etc.)
2. Extract axis labels and data points
3. Note the data ranges and scales
4. Extract any legends or annotations

Return the extracted data in JSON format."""
        }
        
        prompt = prompts.get(analysis_type, prompts["extract_tables"])
        response = await self.analyze_image(document_path, prompt)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"raw_content": response.content}
    
    async def generate_pipeline_from_description(
        self,
        description: str,
        available_connectors: List[str]
    ) -> Dict[str, Any]:
        """Generate pipeline configuration from natural language description"""
        prompt = f"""Convert this natural language description into a data pipeline configuration:

Description: {description}

Available connectors: {', '.join(available_connectors)}

Generate a pipeline configuration with:
1. Clear step-by-step transformations
2. Appropriate connectors for data sources
3. LLM-powered steps where intelligent processing is needed
4. Data validation and quality checks
5. Error handling strategies

Return as JSON with structure:
{{
    "name": "pipeline name",
    "description": "what this pipeline does",
    "steps": [
        {{
            "name": "step name",
            "type": "connector|transform|validate|llm_process",
            "config": {{...}}
        }}
    ],
    "error_handling": {{...}},
    "quality_checks": [...]
}}"""
        
        config = LLMConfig(
            temperature=0.3,
            response_format="json",
            enable_extended_thinking=True,
            max_tokens=8000
        )
        
        response = await self.complete(prompt, config)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.error("Failed to parse pipeline configuration")
            return {}
    
    def _prepare_messages(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for Claude API"""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def _extract_content(self, response) -> str:
        """Extract content from Claude response"""
        if hasattr(response, 'content') and response.content:
            if isinstance(response.content, list):
                # Handle multiple content blocks
                texts = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        texts.append(block.text)
                    elif hasattr(block, 'type') and block.type == 'tool_use':
                        texts.append(json.dumps(block.input))
                return '\n'.join(texts)
            else:
                return response.content[0].text
        return ""
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _get_media_type(self, image_path: str) -> str:
        """Get media type from file extension"""
        ext = Path(image_path).suffix.lower()
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.pdf': 'application/pdf'
        }
        return media_types.get(ext, 'image/jpeg')
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage"""
        pricing = self.PRICING.get(self.model, self.PRICING["claude-sonnet-4-20250514"])
        
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
        
        # Add cache costs if applicable
        if hasattr(usage, 'cache_creation_tokens'):
            cache_write_cost = (usage.cache_creation_tokens / 1_000_000) * pricing["cache_write"]
            cache_read_cost = (usage.cache_read_tokens / 1_000_000) * pricing["cache_read"]
            return input_cost + output_cost + cache_write_cost + cache_read_cost
        
        return input_cost + output_cost
    
    def _generate_cache_key(self, prompt: str, config: LLMConfig) -> str:
        """Generate cache key for prompt and config"""
        import hashlib
        
        key_data = {
            "prompt": prompt,
            "model": self.model,
            "temperature": config.temperature,
            "system_prompt": config.system_prompt,
            "response_format": config.response_format
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"claude_{hashlib.md5(key_str.encode()).hexdigest()}"