"""
Document Representation Classes
Unified structure for representing all document types
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
from pathlib import Path
import hashlib
import json

@dataclass
class DocumentChunk:
    """Represents a chunk of a larger document"""
    content: str
    chunk_index: int
    total_chunks: int
    tokens: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Table:
    """Represents a table extracted from a document"""
    data: pd.DataFrame
    headers: List[str]
    table_index: int
    caption: Optional[str] = None
    source_page: Optional[int] = None

@dataclass
class VisualElement:
    """Represents a visual element (chart, diagram, image)"""
    element_type: str  # 'chart', 'diagram', 'image'
    description: str
    data: Optional[Dict[str, Any]] = None
    source_page: Optional[int] = None
    file_path: Optional[str] = None

@dataclass
class ProcessedDocument:
    """
    Unified representation for all document types
    
    This class provides a consistent interface for working with
    documents regardless of their original format.
    """
    # Basic information
    id: str
    filename: str
    file_path: str
    file_type: str  # 'csv', 'json', 'pdf', 'docx', etc.
    
    # Content
    content: str  # Main text content
    structured_data: Optional[pd.DataFrame] = None  # For CSV/Excel/JSON
    
    # Extracted elements
    tables: List[Table] = field(default_factory=list)
    visual_elements: List[VisualElement] = field(default_factory=list)
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Claude analysis results
    claude_analysis: Optional[Dict[str, Any]] = None
    requires_claude_processing: bool = False
    
    # Timestamps
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Generate ID if not provided"""
        if not self.id:
            # Generate ID from file path and timestamp
            hash_input = f"{self.file_path}:{self.loaded_at.isoformat()}"
            self.id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    @property
    def has_structured_data(self) -> bool:
        """Check if document has structured data (CSV, Excel, JSON)"""
        return self.structured_data is not None and not self.structured_data.empty
    
    @property
    def has_tables(self) -> bool:
        """Check if document has extracted tables"""
        return len(self.tables) > 0
    
    @property
    def has_visual_elements(self) -> bool:
        """Check if document has visual elements"""
        return len(self.visual_elements) > 0
    
    @property
    def is_chunked(self) -> bool:
        """Check if document is split into chunks"""
        return len(self.chunks) > 0
    
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens across all chunks"""
        if self.is_chunked:
            return sum(chunk.tokens for chunk in self.chunks)
        else:
            # Estimate from content length
            return len(self.content) // 4
    
    def get_content_sample(self, max_chars: int = 1000) -> str:
        """Get a sample of the content"""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."
    
    def get_chunk_by_index(self, index: int) -> Optional[DocumentChunk]:
        """Get a specific chunk by index"""
        for chunk in self.chunks:
            if chunk.chunk_index == index:
                return chunk
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_type': self.file_type,
            'content_length': len(self.content),
            'has_structured_data': self.has_structured_data,
            'tables_count': len(self.tables),
            'visual_elements_count': len(self.visual_elements),
            'chunks_count': len(self.chunks),
            'total_tokens': self.total_tokens,
            'metadata': self.metadata,
            'processing_metadata': self.processing_metadata,
            'requires_claude_processing': self.requires_claude_processing,
            'loaded_at': self.loaded_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert document to DataFrame representation
        Useful for pipeline processing
        """
        if self.has_structured_data:
            # Return existing structured data
            return self.structured_data.copy()
        
        # Create DataFrame from content
        data = []
        
        if self.is_chunked:
            # One row per chunk
            for chunk in self.chunks:
                data.append({
                    'document_id': self.id,
                    'filename': self.filename,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'tokens': chunk.tokens,
                    'type': 'chunk'
                })
        else:
            # Single row for entire document
            data.append({
                'document_id': self.id,
                'filename': self.filename,
                'content': self.content,
                'tokens': self.total_tokens,
                'type': 'full_document'
            })
        
        # Add tables as separate rows
        for table in self.tables:
            data.append({
                'document_id': self.id,
                'filename': self.filename,
                'content': table.data.to_string(),
                'type': 'table',
                'table_index': table.table_index
            })
        
        return pd.DataFrame(data)

@dataclass
class DocumentCollection:
    """
    Collection of related documents
    Useful for multi-document processing
    """
    documents: List[ProcessedDocument] = field(default_factory=list)
    collection_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_document(self, doc: ProcessedDocument):
        """Add a document to the collection"""
        self.documents.append(doc)
    
    def get_by_type(self, file_type: str) -> List[ProcessedDocument]:
        """Get all documents of a specific type"""
        return [doc for doc in self.documents if doc.file_type == file_type]
    
    @property
    def total_documents(self) -> int:
        """Total number of documents"""
        return len(self.documents)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens across all documents"""
        return sum(doc.total_tokens for doc in self.documents)
    
    @property
    def document_types(self) -> List[str]:
        """Get unique document types in collection"""
        return list(set(doc.file_type for doc in self.documents))
    
    def to_combined_content(self, separator: str = "\n\n---\n\n") -> str:
        """Combine all document contents"""
        contents = []
        for doc in self.documents:
            contents.append(f"=== {doc.filename} ===\n{doc.content}")
        return separator.join(contents)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert entire collection to DataFrame"""
        dfs = []
        for doc in self.documents:
            df = doc.to_dataframe()
            dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def group_by_similarity(self, threshold: float = 0.8) -> List[List[ProcessedDocument]]:
        """
        Group documents by content similarity
        (Placeholder for future implementation with embeddings)
        """
        # For now, group by file type
        groups = {}
        for doc in self.documents:
            if doc.file_type not in groups:
                groups[doc.file_type] = []
            groups[doc.file_type].append(doc)
        
        return list(groups.values())
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of the collection"""
        return {
            'total_documents': self.total_documents,
            'total_tokens': self.total_tokens,
            'document_types': self.document_types,
            'file_breakdown': {
                file_type: len(self.get_by_type(file_type))
                for file_type in self.document_types
            },
            'has_structured_data': sum(1 for doc in self.documents if doc.has_structured_data),
            'has_tables': sum(1 for doc in self.documents if doc.has_tables),
            'has_visual_elements': sum(1 for doc in self.documents if doc.has_visual_elements),
            'requires_claude_processing': sum(1 for doc in self.documents if doc.requires_claude_processing)
        }