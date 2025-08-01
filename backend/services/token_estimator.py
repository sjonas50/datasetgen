"""
Token Estimation Service for Dataset Generation
Calculates token usage and costs using Anthropic's official API
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import anthropic
from anthropic import AsyncAnthropic
import asyncio
from functools import lru_cache
import hashlib

class TokenEstimator:
    """Estimate tokens and costs for dataset generation"""
    
    # Custom pricing for dataset generation
    PRICING = {
        "claude-sonnet-4-20250514": {
            "input": 200.0 / 1_000_000,   # $200 per million input tokens
            "output": 500.0 / 1_000_000,  # $500 per million output tokens
            "name": "Claude Sonnet 4",
            "max_output_tokens": 8192,
            "supports_vision": True
        },
        "claude-opus-4-20250514": {
            "input": 200.0 / 1_000_000,   # $200 per million input tokens  
            "output": 500.0 / 1_000_000,  # $500 per million output tokens
            "name": "Claude Opus 4",
            "max_output_tokens": 8192,
            "supports_vision": True
        }
    }
    
    # Average tokens per row for different dataset types (more realistic)
    TOKENS_PER_ROW = {
        "qa_pairs": 250,  # Q&A pairs are typically longer
        "classification": 80,   # Text + label
        "ner": 150,  # Sentence + multiple entity annotations
        "summarization": 350,  # Text passage + summary
        "custom": 200,  # Default estimate for custom formats
    }
    
    # Prompt overhead tokens (system prompts, formatting, etc.)
    PROMPT_OVERHEAD = {
        "qa_pairs": 1200,  # Detailed instructions for Q&A generation
        "classification": 800,  # Category definitions and examples
        "ner": 1000,  # Entity type definitions and format specs
        "summarization": 900,  # Length requirements and examples
        "custom": 1500,  # User instructions plus format inference
    }
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            self.client = AsyncAnthropic(api_key=self.api_key)
            print("[TokenEstimator] Initialized with Anthropic API")
        else:
            print("[TokenEstimator] Warning: ANTHROPIC_API_KEY not set. Using estimates.")
            self.client = None
        
        # Cache for token counts (expires after 1 hour)
        self._token_cache = {}
        self._cache_expiry = {}
    
    async def estimate_cost(
        self,
        content: str,
        dataset_type: str,
        target_rows: Optional[int] = None,
        custom_instructions: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514"
    ) -> Dict[str, Any]:
        """
        Estimate tokens and cost for dataset generation
        
        Args:
            content: Combined document content
            dataset_type: Type of dataset to generate
            target_rows: Target number of rows (auto-estimate if None)
            custom_instructions: Additional user instructions
            model: Claude model to use
            
        Returns:
            Dictionary with token counts, costs, and recommendations
        """
        print(f"[TokenEstimator] Estimating for {dataset_type} dataset")
        print(f"[TokenEstimator] Content length: {len(content)} characters")
        
        # Count input tokens
        input_tokens = await self._count_tokens(content, model)
        
        # Add prompt overhead
        prompt_overhead = self.PROMPT_OVERHEAD.get(dataset_type, 300)
        if custom_instructions:
            instructions_tokens = await self._count_tokens(custom_instructions, model)
            prompt_overhead += instructions_tokens
        
        total_input_tokens = input_tokens + prompt_overhead
        
        # Estimate output tokens
        if target_rows is None:
            # Auto-estimate rows based on content size - use character count for better accuracy
            content_chars = len(content)
            target_rows = self._estimate_rows_from_chars(content_chars, dataset_type)
        
        tokens_per_row = self.TOKENS_PER_ROW.get(dataset_type, 100)
        estimated_output_tokens = target_rows * tokens_per_row
        
        # Calculate costs
        pricing = self.PRICING.get(model, self.PRICING["claude-sonnet-4-20250514"])
        input_cost = total_input_tokens * pricing["input"]
        output_cost = estimated_output_tokens * pricing["output"]
        total_cost = input_cost + output_cost
        
        # Estimate processing time (rough estimate)
        estimated_time = self._estimate_processing_time(
            total_input_tokens, estimated_output_tokens
        )
        
        # Generate recommendations
        warnings = []
        recommendations = {}
        
        if total_input_tokens > 100_000:
            warnings.append("Large document detected. Consider using chunk processing.")
            recommendations["optimal_strategy"] = "chunk"
            recommendations["chunk_size"] = 50
            recommendations["reason"] = "Document size exceeds 100K tokens"
        elif total_input_tokens > 50_000:
            recommendations["optimal_strategy"] = "batch"
            recommendations["batch_size"] = 20
            recommendations["reason"] = "Medium-sized document, batch processing recommended"
        else:
            recommendations["optimal_strategy"] = "single"
            recommendations["reason"] = "Document size is optimal for single processing"
        
        if total_cost > 10.0:
            warnings.append(f"Estimated cost exceeds $10 (${total_cost:.2f})")
        
        return {
            "estimation": {
                "total_input_tokens": total_input_tokens,
                "document_tokens": input_tokens,
                "prompt_overhead_tokens": prompt_overhead,
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_rows": target_rows,
                "estimated_time_seconds": estimated_time,
                "tokens_per_row": tokens_per_row,
            },
            "costs": {
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(total_cost, 4),
                "currency": "USD",
                "model": pricing["name"],
            },
            "warnings": warnings,
            "recommendations": recommendations,
            "breakdown": {
                "document_tokens": input_tokens,
                "prompt_tokens": prompt_overhead,
                "tokens_per_row": tokens_per_row,
                "estimated_rows": target_rows,
            }
        }
    
    async def _count_tokens(self, text: str, model: str) -> int:
        """
        Count tokens using Anthropic's API or estimation
        """
        # Check cache first
        cache_key = hashlib.md5(f"{text[:1000]}:{model}".encode()).hexdigest()
        if cache_key in self._token_cache:
            expiry = self._cache_expiry.get(cache_key)
            if expiry and expiry > datetime.utcnow():
                return self._token_cache[cache_key]
        
        if self.client:
            try:
                # Note: Anthropic's count_tokens method is not available in current SDK version
                # Using estimation based on character count for now
                # In future versions, use: response = await self.client.messages.count_tokens(...)
                pass
            except Exception as e:
                print(f"[TokenEstimator] Error counting tokens: {e}")
                # Fall back to estimation
        
        # Rough estimation: ~4 characters per token
        estimated_tokens = len(text) // 4
        return estimated_tokens
    
    def _estimate_rows_from_chars(self, content_chars: int, dataset_type: str) -> int:
        """
        Estimate optimal number of rows based on content character count
        This matches the logic in dataset_generator.py for consistency
        """
        print(f"[TokenEstimator] Estimating rows from {content_chars} characters")
        
        if dataset_type == "qa_pairs":
            # 1 Q&A pair per 100-150 characters (matching dataset_generator.py)
            if content_chars > 10000:
                estimated = max(100, content_chars // 150)
            elif content_chars > 5000:
                estimated = max(50, content_chars // 120)
            elif content_chars > 2000:
                estimated = max(30, content_chars // 100)
            else:
                estimated = max(20, content_chars // 80)
            # Cap at reasonable maximum
            return min(estimated, 5000)
            
        elif dataset_type == "classification":
            # More examples possible - roughly 1 per 80-120 chars
            if content_chars > 10000:
                estimated = max(150, content_chars // 120)
            elif content_chars > 5000:
                estimated = max(80, content_chars // 100)
            else:
                estimated = max(30, content_chars // 80)
            return min(estimated, 5000)
            
        elif dataset_type == "ner":
            # NER needs sentences with entities - 1 per 150-200 chars
            if content_chars > 10000:
                estimated = max(80, content_chars // 200)
            elif content_chars > 5000:
                estimated = max(50, content_chars // 180)
            else:
                estimated = max(20, content_chars // 150)
            return min(estimated, 2000)
            
        elif dataset_type == "summarization":
            # Summarization needs longer chunks - 1 per 300-500 chars
            if content_chars > 10000:
                estimated = max(30, content_chars // 500)
            elif content_chars > 5000:
                estimated = max(20, content_chars // 400)
            else:
                estimated = max(10, content_chars // 300)
            return min(estimated, 1000)
            
        else:
            # Custom/default - use moderate scaling
            if content_chars > 10000:
                estimated = max(80, content_chars // 150)
            elif content_chars > 5000:
                estimated = max(50, content_chars // 130)
            else:
                estimated = max(20, content_chars // 100)
            return min(estimated, 3000)
    
    def _estimate_rows(self, content_tokens: int, dataset_type: str) -> int:
        """
        Estimate optimal number of rows based on content size
        More realistic estimation based on content density
        """
        # Base estimation on content density - much more aggressive scaling
        if dataset_type == "qa_pairs":
            # Roughly 1 Q&A pair per 50-100 tokens of content
            # For a typical document, this generates many more pairs
            if content_tokens > 10000:
                return min(max(content_tokens // 100, 100), 2000)
            elif content_tokens > 5000:
                return min(max(content_tokens // 80, 50), 1000)
            else:
                return min(max(content_tokens // 50, 20), 500)
        elif dataset_type == "classification":
            # More examples possible from same content
            if content_tokens > 10000:
                return min(max(content_tokens // 75, 150), 3000)
            elif content_tokens > 5000:
                return min(max(content_tokens // 60, 100), 1500)
            else:
                return min(max(content_tokens // 40, 30), 750)
        elif dataset_type == "ner":
            # NER depends on entity density
            if content_tokens > 10000:
                return min(max(content_tokens // 150, 80), 1000)
            elif content_tokens > 5000:
                return min(max(content_tokens // 120, 50), 500)
            else:
                return min(max(content_tokens // 80, 20), 250)
        elif dataset_type == "summarization":
            # Fewer but longer examples
            if content_tokens > 10000:
                return min(max(content_tokens // 300, 30), 500)
            elif content_tokens > 5000:
                return min(max(content_tokens // 250, 20), 250)
            else:
                return min(max(content_tokens // 200, 10), 125)
        else:
            # Default estimation
            if content_tokens > 10000:
                return min(max(content_tokens // 120, 80), 1500)
            elif content_tokens > 5000:
                return min(max(content_tokens // 100, 50), 750)
            else:
                return min(max(content_tokens // 80, 20), 400)
    
    def _estimate_processing_time(self, input_tokens: int, output_tokens: int) -> int:
        """
        Estimate processing time in seconds
        """
        # Rough estimates based on token counts
        # Claude can process ~10K tokens/second input, generate ~1K tokens/second
        input_time = input_tokens / 10_000
        output_time = output_tokens / 1_000
        
        # Add overhead for API calls, network, etc.
        overhead = 5
        
        return int(input_time + output_time + overhead)
    
    async def estimate_multi_document_cost(
        self,
        documents: List[Dict[str, Any]],
        dataset_type: str,
        processing_strategy: str = "auto",
        model: str = "claude-sonnet-4-20250514"
    ) -> Dict[str, Any]:
        """
        Estimate cost for processing multiple documents
        """
        total_tokens = 0
        document_details = []
        
        for doc in documents:
            content = doc.get("content", "")
            tokens = await self._count_tokens(content, model)
            total_tokens += tokens
            
            document_details.append({
                "filename": doc.get("filename", "Unknown"),
                "tokens": tokens,
                "size_mb": len(content) / (1024 * 1024),
            })
        
        # Determine optimal strategy
        if processing_strategy == "auto":
            if total_tokens > 150_000:
                processing_strategy = "chunk"
            elif len(documents) > 10:
                processing_strategy = "batch"
            else:
                processing_strategy = "combine"
        
        # Calculate chunks/batches needed
        chunks_needed = 1
        if processing_strategy == "chunk":
            chunk_size = 50_000  # tokens per chunk
            chunks_needed = (total_tokens + chunk_size - 1) // chunk_size
        elif processing_strategy == "batch":
            batch_size = 5  # documents per batch
            chunks_needed = (len(documents) + batch_size - 1) // batch_size
        
        # Estimate total cost
        base_estimate = await self.estimate_cost(
            "x" * (total_tokens * 4),  # Approximate character count
            dataset_type,
            model=model
        )
        
        # Adjust for processing strategy
        if processing_strategy in ["chunk", "batch"]:
            # Add overhead for multiple API calls
            overhead_multiplier = 1.1
            base_estimate["costs"]["total_cost"] *= overhead_multiplier * chunks_needed
            base_estimate["estimation"]["estimated_time_seconds"] *= chunks_needed
        
        return {
            **base_estimate,
            "multi_document": {
                "total_documents": len(documents),
                "total_tokens": total_tokens,
                "processing_strategy": processing_strategy,
                "chunks_needed": chunks_needed,
                "document_details": document_details,
            }
        }

# Global instance
token_estimator = TokenEstimator()