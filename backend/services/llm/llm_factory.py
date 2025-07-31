from typing import Optional, Dict, Any
from enum import Enum

from .base import BaseLLMService, LLMProvider
from .claude_service import ClaudeService
from .openai_service import OpenAIService
from .gemini_service import GeminiService
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Factory for creating LLM service instances"""
    
    _instances: Dict[str, BaseLLMService] = {}
    
    @classmethod
    def create(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        use_singleton: bool = True
    ) -> BaseLLMService:
        """
        Create or get an LLM service instance
        
        Args:
            provider: LLM provider (claude, openai, gemini)
            model: Specific model to use (optional)
            api_key: API key (optional, will use from settings)
            use_singleton: Whether to reuse existing instances
        
        Returns:
            LLM service instance
        """
        # Normalize provider name
        provider = provider.lower()
        
        # Create cache key
        cache_key = f"{provider}:{model or 'default'}"
        
        # Return existing instance if singleton mode
        if use_singleton and cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Create new instance
        service = cls._create_instance(provider, model, api_key)
        
        # Cache if singleton mode
        if use_singleton:
            cls._instances[cache_key] = service
        
        return service
    
    @classmethod
    def _create_instance(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> BaseLLMService:
        """Create a new LLM service instance"""
        
        if provider == LLMProvider.CLAUDE.value:
            return ClaudeService(
                api_key=api_key or settings.ANTHROPIC_API_KEY,
                model=model or "sonnet-4"
            )
        
        elif provider == LLMProvider.OPENAI.value:
            return OpenAIService(
                api_key=api_key or settings.OPENAI_API_KEY,
                model=model or "gpt-4o"
            )
        
        elif provider == LLMProvider.GEMINI.value:
            return GeminiService(
                api_key=api_key or settings.GOOGLE_API_KEY,
                model=model or "gemini-2.5-pro"
            )
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    @classmethod
    def get_best_for_task(cls, task_type: str) -> BaseLLMService:
        """
        Get the best LLM service for a specific task
        
        Args:
            task_type: Type of task (schema_inference, code_generation, etc.)
        
        Returns:
            Most suitable LLM service
        """
        task_recommendations = {
            "schema_inference": ("claude", "sonnet-4"),  # Extended thinking
            "code_generation": ("claude", "sonnet-4"),    # Best coding performance
            "vision_analysis": ("claude", "sonnet-4"),    # Strong vision capabilities
            "long_context": ("gemini", "gemini-2.5-pro"), # 2M+ context window
            "cost_sensitive": ("gemini", "gemini-2.5-flash"), # Cheapest option
            "reasoning": ("openai", "o3-mini"),          # Reasoning tasks
            "general": ("claude", "sonnet-4")            # Best overall
        }
        
        provider, model = task_recommendations.get(
            task_type, 
            task_recommendations["general"]
        )
        
        try:
            return cls.create(provider, model)
        except Exception as e:
            logger.warning(f"Failed to create {provider} service: {str(e)}")
            # Fallback to any available service
            return cls._get_any_available()
    
    @classmethod
    def _get_any_available(cls) -> BaseLLMService:
        """Get any available LLM service as fallback"""
        # Try in order of preference
        providers = [
            (LLMProvider.CLAUDE, settings.ANTHROPIC_API_KEY),
            (LLMProvider.OPENAI, settings.OPENAI_API_KEY),
            (LLMProvider.GEMINI, settings.GOOGLE_API_KEY)
        ]
        
        for provider, api_key in providers:
            if api_key:
                try:
                    return cls.create(provider.value)
                except Exception:
                    continue
        
        raise RuntimeError("No LLM service available. Please configure at least one API key.")
    
    @classmethod
    def get_usage_stats(cls) -> Dict[str, Any]:
        """Get usage statistics for all active instances"""
        stats = {}
        
        for key, service in cls._instances.items():
            stats[key] = service.get_usage_stats()
        
        # Calculate totals
        total_tokens = sum(s["total_tokens"] for s in stats.values())
        total_cost = sum(s["total_cost"] for s in stats.values())
        total_requests = sum(s["request_count"] for s in stats.values())
        
        return {
            "services": stats,
            "totals": {
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "total_requests": total_requests
            }
        }
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached instances"""
        cls._instances.clear()