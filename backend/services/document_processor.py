"""
Document Processing Service for Multi-Document and Large Dataset Handling
Implements intelligent chunking and batching strategies
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime
import hashlib
import math

class DocumentProcessor:
    """Process and chunk documents for efficient dataset generation"""
    
    # Token limits for different processing strategies
    TOKEN_LIMITS = {
        "single": 50_000,      # Process as single document
        "batch": 100_000,      # Process in batches
        "chunk": 150_000,      # Chunk into smaller pieces
    }
    
    # Chunk sizes for different strategies
    CHUNK_SIZES = {
        "small": 25_000,       # Small chunks for detailed processing
        "medium": 50_000,      # Medium chunks for balanced processing
        "large": 75_000,       # Large chunks for faster processing
    }
    
    def __init__(self):
        self.max_context_window = 200_000  # Claude's context window
        self.safety_margin = 0.8  # Use only 80% of context window
        
    def analyze_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a collection of documents to determine optimal processing strategy
        
        Args:
            documents: List of document dictionaries with content and metadata
            
        Returns:
            Analysis results with recommended strategy
        """
        total_size = 0
        total_tokens = 0
        file_stats = []
        
        for doc in documents:
            content = doc.get("content", "")
            tokens = doc.get("tokens", len(content) // 4)  # Rough estimate if not provided
            size_mb = len(content) / (1024 * 1024)
            
            total_size += size_mb
            total_tokens += tokens
            
            file_stats.append({
                "filename": doc.get("filename", "Unknown"),
                "tokens": tokens,
                "size_mb": round(size_mb, 2),
                "type": doc.get("file_type", "unknown"),
            })
        
        # Determine optimal strategy
        strategy = self._determine_strategy(total_tokens, len(documents))
        
        # Calculate chunks needed
        chunks_needed = self._calculate_chunks(total_tokens, strategy)
        
        return {
            "total_documents": len(documents),
            "total_size_mb": round(total_size, 2),
            "total_tokens": total_tokens,
            "recommended_strategy": strategy,
            "chunks_needed": chunks_needed,
            "estimated_api_calls": chunks_needed,
            "file_statistics": file_stats,
            "warnings": self._generate_warnings(total_tokens, len(documents)),
        }
    
    def chunk_documents(
        self, 
        documents: List[Dict[str, Any]], 
        strategy: str = "auto",
        chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk documents according to the specified strategy
        
        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy ("auto", "combine", "chunk", "batch")
            chunk_size: Optional custom chunk size in tokens
            
        Returns:
            List of document chunks ready for processing
        """
        if strategy == "auto":
            total_tokens = sum(doc.get("tokens", len(doc.get("content", "")) // 4) for doc in documents)
            strategy = self._determine_strategy(total_tokens, len(documents))
        
        if strategy == "single" or strategy == "combine":
            return self._combine_documents(documents, chunk_size)
        elif strategy == "chunk":
            return self._chunk_large_documents(documents, chunk_size)
        elif strategy == "batch":
            return self._batch_documents(documents, chunk_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _determine_strategy(self, total_tokens: int, num_documents: int) -> str:
        """Determine optimal processing strategy based on document characteristics"""
        
        # Single document or small total
        if num_documents == 1 or total_tokens < self.TOKEN_LIMITS["single"]:
            return "single"
        
        # Many small documents - batch them
        if num_documents > 10 and total_tokens / num_documents < 5000:
            return "batch"
        
        # Large total size - chunk it
        if total_tokens > self.TOKEN_LIMITS["chunk"]:
            return "chunk"
        
        # Medium size - try to combine
        return "combine"
    
    def _combine_documents(
        self, 
        documents: List[Dict[str, Any]], 
        max_chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple small documents into larger chunks
        """
        if max_chunk_size is None:
            max_chunk_size = int(self.max_context_window * self.safety_margin)
        
        chunks = []
        current_chunk = {
            "documents": [],
            "content": "",
            "tokens": 0,
            "source_files": [],
        }
        
        for doc in documents:
            doc_tokens = doc.get("tokens", len(doc.get("content", "")) // 4)
            
            # If adding this doc would exceed limit, start new chunk
            if current_chunk["tokens"] + doc_tokens > max_chunk_size and current_chunk["documents"]:
                chunks.append(current_chunk)
                current_chunk = {
                    "documents": [],
                    "content": "",
                    "tokens": 0,
                    "source_files": [],
                }
            
            # Add document to current chunk
            current_chunk["documents"].append(doc)
            current_chunk["content"] += f"\n\n--- Document: {doc.get('filename', 'Unknown')} ---\n\n"
            current_chunk["content"] += doc.get("content", "")
            current_chunk["tokens"] += doc_tokens
            current_chunk["source_files"].append(doc.get("filename", "Unknown"))
        
        # Add final chunk
        if current_chunk["documents"]:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_large_documents(
        self, 
        documents: List[Dict[str, Any]], 
        chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Split large documents into smaller chunks
        """
        if chunk_size is None:
            chunk_size = self.CHUNK_SIZES["medium"]
        
        chunks = []
        
        for doc in documents:
            content = doc.get("content", "")
            doc_tokens = doc.get("tokens", len(content) // 4)
            
            if doc_tokens <= chunk_size:
                # Document fits in single chunk
                chunks.append({
                    "documents": [doc],
                    "content": content,
                    "tokens": doc_tokens,
                    "source_files": [doc.get("filename", "Unknown")],
                    "chunk_index": 0,
                    "total_chunks": 1,
                })
            else:
                # Split document into chunks
                # Try to split at natural boundaries (paragraphs)
                paragraphs = content.split("\n\n")
                
                current_chunk = {
                    "content": "",
                    "tokens": 0,
                    "chunk_index": 0,
                }
                chunk_count = 0
                
                for para in paragraphs:
                    para_tokens = len(para) // 4
                    
                    if current_chunk["tokens"] + para_tokens > chunk_size and current_chunk["content"]:
                        # Save current chunk
                        chunks.append({
                            "documents": [doc],
                            "content": current_chunk["content"],
                            "tokens": current_chunk["tokens"],
                            "source_files": [doc.get("filename", "Unknown")],
                            "chunk_index": chunk_count,
                            "total_chunks": -1,  # Will update later
                        })
                        chunk_count += 1
                        
                        # Start new chunk
                        current_chunk = {
                            "content": para,
                            "tokens": para_tokens,
                        }
                    else:
                        # Add to current chunk
                        if current_chunk["content"]:
                            current_chunk["content"] += "\n\n"
                        current_chunk["content"] += para
                        current_chunk["tokens"] += para_tokens
                
                # Add final chunk
                if current_chunk["content"]:
                    chunks.append({
                        "documents": [doc],
                        "content": current_chunk["content"],
                        "tokens": current_chunk["tokens"],
                        "source_files": [doc.get("filename", "Unknown")],
                        "chunk_index": chunk_count,
                        "total_chunks": chunk_count + 1,
                    })
                
                # Update total chunks count
                doc_chunks = [c for c in chunks if doc.get("filename") in c["source_files"]]
                for chunk in doc_chunks:
                    chunk["total_chunks"] = len(doc_chunks)
        
        return chunks
    
    def _batch_documents(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch documents for parallel processing
        """
        if batch_size is None:
            batch_size = 5  # Default batch size
        
        batches = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Combine batch into single chunk
            combined_content = ""
            total_tokens = 0
            source_files = []
            
            for doc in batch_docs:
                combined_content += f"\n\n--- Document: {doc.get('filename', 'Unknown')} ---\n\n"
                combined_content += doc.get("content", "")
                total_tokens += doc.get("tokens", len(doc.get("content", "")) // 4)
                source_files.append(doc.get("filename", "Unknown"))
            
            batches.append({
                "documents": batch_docs,
                "content": combined_content,
                "tokens": total_tokens,
                "source_files": source_files,
                "batch_index": i // batch_size,
                "total_batches": math.ceil(len(documents) / batch_size),
            })
        
        return batches
    
    def _calculate_chunks(self, total_tokens: int, strategy: str) -> int:
        """Calculate number of chunks needed for given strategy"""
        if strategy == "single":
            return 1
        elif strategy == "chunk":
            chunk_size = self.CHUNK_SIZES["medium"]
            return math.ceil(total_tokens / chunk_size)
        elif strategy == "batch":
            # Rough estimate - 5 docs per batch
            return math.ceil(total_tokens / (self.TOKEN_LIMITS["single"] * 0.8))
        else:
            return 1
    
    def _generate_warnings(self, total_tokens: int, num_documents: int) -> List[str]:
        """Generate warnings based on document analysis"""
        warnings = []
        
        if total_tokens > 500_000:
            warnings.append(f"Very large dataset ({total_tokens:,} tokens). Processing will be split into multiple chunks.")
        
        if num_documents > 50:
            warnings.append(f"Large number of documents ({num_documents}). Consider batch processing for efficiency.")
        
        if total_tokens > 1_000_000:
            warnings.append("Dataset size may result in significant processing costs. Consider sampling or filtering.")
        
        return warnings
    
    def combine_for_context(
        self, 
        documents: List[Dict[str, Any]], 
        max_tokens: int = 50_000
    ) -> List[Dict[str, Any]]:
        """
        Intelligently combine related documents for context
        
        This method tries to group related documents together
        to preserve context during processing.
        """
        # For now, simple implementation - can be enhanced with similarity measures
        return self._combine_documents(documents, max_tokens)
    
    def estimate_processing_chunks(
        self, 
        documents: List[Dict[str, Any]], 
        dataset_type: str
    ) -> Dict[str, Any]:
        """
        Estimate how many processing chunks will be needed
        """
        analysis = self.analyze_documents(documents)
        
        # Adjust based on dataset type
        overhead_multiplier = {
            "qa_pairs": 1.2,
            "classification": 1.1,
            "ner": 1.3,
            "summarization": 1.4,
            "custom": 1.2,
        }.get(dataset_type, 1.2)
        
        adjusted_chunks = math.ceil(analysis["chunks_needed"] * overhead_multiplier)
        
        return {
            "base_chunks": analysis["chunks_needed"],
            "adjusted_chunks": adjusted_chunks,
            "strategy": analysis["recommended_strategy"],
            "reason": f"Dataset type '{dataset_type}' requires {overhead_multiplier}x overhead",
        }

# Global instance
document_processor = DocumentProcessor()