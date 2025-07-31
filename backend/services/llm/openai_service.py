import openai
from openai import AsyncOpenAI
from typing import Dict, Any, Optional, List
import base64
import json
from pathlib import Path

from .base import BaseLLMService, LLMResponse, LLMConfig
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class OpenAIService(BaseLLMService):
    """OpenAI GPT-4o implementation"""
    
    MODELS = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4-turbo-preview",
        "o3": "o3",
        "o3-mini": "o3-mini"
    }
    
    PRICING = {
        "gpt-4o": {
            "input": 2.5,    # $2.50 per million
            "output": 10.0   # $10 per million
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        api_key = api_key or settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        model_name = self.MODELS.get(model, model)
        super().__init__(api_key, model_name)
        
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def complete(
        self, 
        prompt: str, 
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion"""
        config = config or LLMConfig()
        
        try:
            messages = self._prepare_messages(prompt, config.system_prompt)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens or 4096,
                "top_p": config.top_p,
            }
            
            # Add response format if specified
            if config.response_format == "json":
                request_params["response_format"] = {"type": "json_object"}
            
            # Add tools if provided
            if config.tools:
                request_params["tools"] = config.tools
                if config.tool_choice:
                    request_params["tool_choice"] = config.tool_choice
            
            # Make API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract content
            content = response.choices[0].message.content
            
            # Handle tool calls
            if response.choices[0].message.tool_calls:
                tool_results = []
                for tool_call in response.choices[0].message.tool_calls:
                    tool_results.append({
                        "tool": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    })
                content = json.dumps(tool_results)
            
            # Calculate usage
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": self._calculate_cost(response.usage)
            }
            
            self._update_usage(usage["total_tokens"], usage["cost"])
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion with function calling"""
        config = config or LLMConfig()
        config.tools = tools
        return await self.complete(prompt, config)
    
    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Analyze image with GPT-4 Vision"""
        config = config or LLMConfig()
        
        # Encode image
        base64_image = self._encode_image(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",  # Vision-enabled model
                messages=messages,
                max_tokens=config.max_tokens or 4096,
                temperature=config.temperature
            )
            
            content = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": self._calculate_cost(response.usage)
            }
            
            return LLMResponse(
                content=content,
                model="gpt-4o",
                usage=usage,
                metadata={"vision_analysis": True}
            )
            
        except Exception as e:
            logger.error(f"Vision analysis error: {str(e)}")
            raise
    
    async def use_reasoning_model(
        self,
        prompt: str,
        reasoning_model: str = "o3-mini"
    ) -> LLMResponse:
        """Use O3 reasoning model for complex tasks"""
        config = LLMConfig(
            temperature=0.2,  # Lower temperature for reasoning
            max_tokens=8000   # More tokens for detailed reasoning
        )
        
        # Switch to reasoning model temporarily
        original_model = self.model
        self.model = self.MODELS.get(reasoning_model, reasoning_model)
        
        try:
            response = await self.complete(prompt, config)
            return response
        finally:
            self.model = original_model
    
    def _prepare_messages(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
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
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage"""
        pricing = self.PRICING.get(self.model, self.PRICING["gpt-4o"])
        
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost