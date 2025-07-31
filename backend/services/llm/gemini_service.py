import google.generativeai as genai
from typing import Dict, Any, Optional, List
import json
from pathlib import Path

from .base import BaseLLMService, LLMResponse, LLMConfig
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class GeminiService(BaseLLMService):
    """Google Gemini implementation"""
    
    MODELS = {
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash"
    }
    
    PRICING = {
        "gemini-2.5-pro": {
            "input": 0.50,   # $0.50 per million
            "output": 1.50   # $1.50 per million
        },
        "gemini-2.5-flash": {
            "input": 0.075,
            "output": 0.30
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-pro"):
        api_key = api_key or settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("Google API key not provided")
        
        model_name = self.MODELS.get(model, model)
        super().__init__(api_key, model_name)
        
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model_name)
    
    async def complete(
        self, 
        prompt: str, 
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion"""
        config = config or LLMConfig()
        
        try:
            # Prepare generation config
            generation_config = genai.GenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_tokens or 4096,
            )
            
            # Add response format if JSON
            if config.response_format == "json":
                generation_config.response_mime_type = "application/json"
            
            # Build full prompt with system message if provided
            full_prompt = prompt
            if config.system_prompt:
                full_prompt = f"{config.system_prompt}\n\n{prompt}"
            
            # Generate response
            response = await self.model_instance.generate_content_async(
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract content
            content = response.text
            
            # Estimate token usage (Gemini doesn't provide exact counts)
            estimated_tokens = self._estimate_tokens(full_prompt, content)
            
            usage = {
                "input_tokens": estimated_tokens["input"],
                "output_tokens": estimated_tokens["output"],
                "total_tokens": estimated_tokens["total"],
                "cost": self._calculate_cost_from_estimates(estimated_tokens)
            }
            
            self._update_usage(usage["total_tokens"], usage["cost"])
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata={
                    "safety_ratings": self._extract_safety_ratings(response)
                }
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise
    
    async def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion with function calling"""
        # Convert tools to Gemini format
        gemini_tools = self._convert_tools_to_gemini_format(tools)
        
        config = config or LLMConfig()
        
        try:
            # Create model with tools
            model_with_tools = genai.GenerativeModel(
                self.model,
                tools=gemini_tools
            )
            
            response = await model_with_tools.generate_content_async(prompt)
            
            # Extract function calls if any
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call'):
                        # Format function call response
                        content = json.dumps({
                            "function": part.function_call.name,
                            "arguments": dict(part.function_call.args)
                        })
                        break
                else:
                    content = response.text
            else:
                content = response.text
            
            estimated_tokens = self._estimate_tokens(prompt, content)
            usage = {
                "input_tokens": estimated_tokens["input"],
                "output_tokens": estimated_tokens["output"],
                "total_tokens": estimated_tokens["total"],
                "cost": self._calculate_cost_from_estimates(estimated_tokens)
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata={"tools_used": True}
            )
            
        except Exception as e:
            logger.error(f"Gemini tools error: {str(e)}")
            raise
    
    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Analyze image with Gemini Vision"""
        config = config or LLMConfig()
        
        try:
            # Upload image
            image_file = genai.upload_file(image_path)
            
            # Generate response with image
            response = await self.model_instance.generate_content_async([
                prompt,
                image_file
            ])
            
            content = response.text
            estimated_tokens = self._estimate_tokens(prompt, content)
            
            usage = {
                "input_tokens": estimated_tokens["input"] + 258,  # Add tokens for image
                "output_tokens": estimated_tokens["output"],
                "total_tokens": estimated_tokens["total"] + 258,
                "cost": self._calculate_cost_from_estimates({
                    "input": estimated_tokens["input"] + 258,
                    "output": estimated_tokens["output"]
                })
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata={
                    "vision_analysis": True,
                    "image_path": image_path
                }
            )
            
        except Exception as e:
            logger.error(f"Gemini vision error: {str(e)}")
            raise
    
    async def analyze_long_context(
        self,
        documents: List[str],
        prompt: str
    ) -> LLMResponse:
        """Leverage Gemini's 2M+ token context window"""
        # Combine all documents
        combined_context = "\n\n---DOCUMENT SEPARATOR---\n\n".join(documents)
        
        full_prompt = f"""Context Documents:
{combined_context}

Task:
{prompt}"""
        
        config = LLMConfig(
            temperature=0.3,
            max_tokens=8000
        )
        
        return await self.complete(full_prompt, config)
    
    def _convert_tools_to_gemini_format(self, tools: List[Dict[str, Any]]) -> List:
        """Convert OpenAI-style tools to Gemini format"""
        gemini_tools = []
        
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                gemini_tool = {
                    "function_declarations": [{
                        "name": func["name"],
                        "description": func["description"],
                        "parameters": func.get("parameters", {})
                    }]
                }
                gemini_tools.append(gemini_tool)
        
        return gemini_tools
    
    def _estimate_tokens(self, input_text: str, output_text: str) -> Dict[str, int]:
        """Estimate token counts (Gemini doesn't provide exact counts)"""
        # Rough estimation: 1 token ≈ 4 characters
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4
        
        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        }
    
    def _calculate_cost_from_estimates(self, token_estimates: Dict[str, int]) -> float:
        """Calculate cost from token estimates"""
        pricing = self.PRICING.get(self.model, self.PRICING["gemini-2.5-pro"])
        
        input_cost = (token_estimates["input"] / 1_000_000) * pricing["input"]
        output_cost = (token_estimates["output"] / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _extract_safety_ratings(self, response) -> List[Dict[str, str]]:
        """Extract safety ratings from response"""
        safety_ratings = []
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'safety_ratings'):
                for rating in candidate.safety_ratings:
                    safety_ratings.append({
                        "category": rating.category.name,
                        "probability": rating.probability.name
                    })
        
        return safety_ratings