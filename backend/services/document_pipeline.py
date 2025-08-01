"""
Document Processing Pipeline
Orchestrates the entire document processing workflow
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import pandas as pd

from services.document_loader import UnifiedDocumentLoader, DocumentLoadError
from services.document import ProcessedDocument, DocumentCollection, DocumentChunk, Table, VisualElement
from services.claude_service import ClaudeService
from services.document_processor import DocumentProcessor
from services.document_extractor import DocumentExtractor

class DocumentPipeline:
    """
    Main orchestrator for document processing
    
    Handles:
    1. Loading documents with error recovery
    2. Processing with Claude's vision API when needed
    3. Chunking and batching strategies
    4. Progress tracking and caching
    """
    
    def __init__(
        self, 
        claude_service: Optional[ClaudeService] = None,
        enable_caching: bool = True
    ):
        self.claude_service = claude_service or ClaudeService()
        self.document_loader = UnifiedDocumentLoader(self.claude_service)
        self.document_processor = DocumentProcessor()
        self.document_extractor = DocumentExtractor()
        self.enable_caching = enable_caching
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        
    def add_progress_callback(self, callback: Callable):
        """Add a callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    def _update_progress(self, stage: str, current: int, total: int, message: str = ""):
        """Update progress through callbacks"""
        progress_data = {
            'stage': stage,
            'current': current,
            'total': total,
            'percentage': (current / total * 100) if total > 0 else 0,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for callback in self.progress_callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                print(f"[DocumentPipeline] Progress callback error: {e}")
    
    async def process_single_document(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        options: Dict[str, Any] = None
    ) -> ProcessedDocument:
        """
        Process a single document through the entire pipeline
        
        Args:
            file_path: Path to the document
            file_type: Optional MIME type
            options: Processing options
            
        Returns:
            ProcessedDocument with all extracted information
        """
        file_path = Path(file_path)
        options = options or {}
        
        print(f"[DocumentPipeline] Processing {file_path.name}")
        
        # Stage 1: Load document
        self._update_progress("loading", 0, 1, f"Loading {file_path.name}")
        
        try:
            loaded_doc = await self.document_loader.load_document(
                file_path, file_type, options
            )
        except DocumentLoadError as e:
            print(f"[DocumentPipeline] Failed to load {file_path.name}: {e}")
            raise
        
        # Create ProcessedDocument
        doc = ProcessedDocument(
            id="",  # Will be auto-generated
            filename=file_path.name,
            file_path=str(file_path),
            file_type=loaded_doc['format'],
            content=loaded_doc['content'],
            structured_data=loaded_doc.get('structured_data'),
            metadata=loaded_doc.get('metadata', {}),
            processing_metadata={
                'processing_method': loaded_doc.get('processing_method'),
                'warnings': loaded_doc.get('warnings', [])
            }
        )
        
        # Stage 2: Claude processing if needed
        if loaded_doc.get('requires_claude_processing') and self.claude_service.enabled:
            self._update_progress("processing", 0, 1, f"Processing with Claude Vision")
            
            try:
                if doc.file_type == 'pdf':
                    # Use native PDF processing
                    claude_result = await self._process_pdf_with_claude(doc)
                elif doc.file_type in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                    # Use vision API for images
                    claude_result = await self._process_image_with_claude(doc)
                else:
                    claude_result = None
                
                if claude_result:
                    doc.claude_analysis = claude_result
                    doc.processed_at = datetime.utcnow()
                    
                    # Extract tables and visual elements from Claude analysis
                    self._extract_structured_elements(doc, claude_result)
                    
            except Exception as e:
                print(f"[DocumentPipeline] Claude processing error: {e}")
                doc.processing_metadata['claude_error'] = str(e)
        
        # Stage 3: Apply chunking if needed
        if options.get('enable_chunking') and len(doc.content) > 10000:
            self._update_progress("chunking", 0, 1, "Applying chunking strategy")
            
            chunks = self._chunk_document(doc, options.get('chunk_size', 50000))
            doc.chunks = chunks
        
        self._update_progress("complete", 1, 1, f"Completed {file_path.name}")
        
        return doc
    
    async def process_multiple_documents(
        self,
        file_paths: List[Union[str, Path]],
        options: Dict[str, Any] = None
    ) -> DocumentCollection:
        """
        Process multiple documents with intelligent batching
        
        Args:
            file_paths: List of document paths
            options: Processing options
            
        Returns:
            DocumentCollection with all processed documents
        """
        options = options or {}
        collection = DocumentCollection()
        
        total_files = len(file_paths)
        print(f"[DocumentPipeline] Processing {total_files} documents")
        
        # Determine processing strategy
        if total_files > 10 and options.get('parallel_processing', True):
            # Process in batches
            batch_size = options.get('batch_size', 5)
            documents = await self._process_in_batches(file_paths, batch_size, options)
        else:
            # Process sequentially
            documents = []
            for i, file_path in enumerate(file_paths):
                self._update_progress("processing", i, total_files, f"Processing {Path(file_path).name}")
                
                try:
                    doc = await self.process_single_document(file_path, options=options)
                    documents.append(doc)
                except Exception as e:
                    print(f"[DocumentPipeline] Error processing {file_path}: {e}")
                    # Continue with other documents
                    continue
        
        # Add to collection
        for doc in documents:
            collection.add_document(doc)
        
        # Analyze collection
        collection.collection_metadata = {
            'processing_summary': collection.get_processing_summary(),
            'processing_timestamp': datetime.utcnow().isoformat(),
            'options_used': options
        }
        
        return collection
    
    async def _process_in_batches(
        self,
        file_paths: List[Union[str, Path]],
        batch_size: int,
        options: Dict[str, Any]
    ) -> List[ProcessedDocument]:
        """Process documents in batches for efficiency"""
        documents = []
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(file_paths) + batch_size - 1) // batch_size
            
            self._update_progress(
                "batch_processing", 
                batch_num, 
                total_batches, 
                f"Processing batch {batch_num}/{total_batches}"
            )
            
            # Process batch in parallel
            tasks = []
            for file_path in batch:
                task = self.process_single_document(file_path, options=options)
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            for result in batch_results:
                if isinstance(result, ProcessedDocument):
                    documents.append(result)
                else:
                    print(f"[DocumentPipeline] Batch processing error: {result}")
        
        return documents
    
    async def _process_pdf_with_claude(self, doc: ProcessedDocument) -> Dict[str, Any]:
        """Process PDF using Claude's native PDF support"""
        # Read PDF file for Claude
        pdf_path = Path(doc.file_path)
        
        # Create messages for Claude with PDF
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_path": str(pdf_path)
                    }
                },
                {
                    "type": "text",
                    "text": """Analyze this PDF document and extract:
1. Document type and purpose
2. All tables with their structure and data
3. Key sections and headings
4. Visual elements (charts, diagrams) with descriptions
5. Important metrics, dates, or values
6. Overall summary

Format as JSON with sections for each type of content."""
                }
            ]
        }]
        
        response = await self.claude_service.client.messages.create(
            model=self.claude_service.model,
            max_tokens=self.claude_service.max_tokens,
            messages=messages
        )
        
        # Parse response
        try:
            import json
            import re
            
            content = response.content[0].text
            json_match = re.search(r'\{[\s\S]*\}', content)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return {'raw_analysis': content}
                
        except Exception as e:
            print(f"[DocumentPipeline] Error parsing Claude response: {e}")
            return {'raw_analysis': response.content[0].text}
    
    async def _process_image_with_claude(self, doc: ProcessedDocument) -> Dict[str, Any]:
        """Process image using Claude's vision API"""
        # Use existing image extraction method
        result = await self.claude_service.extract_from_image(Path(doc.file_path))
        return result
    
    def _extract_structured_elements(self, doc: ProcessedDocument, claude_analysis: Dict[str, Any]):
        """Extract tables and visual elements from Claude analysis"""
        # Extract tables
        if 'tables' in claude_analysis:
            for i, table_data in enumerate(claude_analysis['tables']):
                if isinstance(table_data, dict) and 'data' in table_data:
                    # Convert to DataFrame
                    try:
                        df = pd.DataFrame(table_data['data'])
                        table = Table(
                            data=df,
                            headers=table_data.get('headers', list(df.columns)),
                            table_index=i,
                            caption=table_data.get('caption'),
                            source_page=table_data.get('page')
                        )
                        doc.tables.append(table)
                    except Exception as e:
                        print(f"[DocumentPipeline] Error creating table: {e}")
        
        # Extract visual elements
        if 'visual_elements' in claude_analysis:
            for i, element_data in enumerate(claude_analysis['visual_elements']):
                if isinstance(element_data, dict):
                    visual = VisualElement(
                        element_type=element_data.get('type', 'unknown'),
                        description=element_data.get('description', ''),
                        data=element_data.get('data'),
                        source_page=element_data.get('page')
                    )
                    doc.visual_elements.append(visual)
    
    def _chunk_document(self, doc: ProcessedDocument, chunk_size: int) -> List[DocumentChunk]:
        """Apply intelligent chunking to document"""
        chunks = []
        content = doc.content
        
        # Use document processor's chunking logic
        doc_dict = {
            'filename': doc.filename,
            'content': content,
            'tokens': len(content) // 4,
            'file_type': doc.file_type
        }
        
        chunked_docs = self.document_processor._chunk_large_documents([doc_dict], chunk_size)
        
        # Convert to DocumentChunk objects
        for chunk_data in chunked_docs:
            chunk = DocumentChunk(
                content=chunk_data['content'],
                chunk_index=chunk_data.get('chunk_index', 0),
                total_chunks=chunk_data.get('total_chunks', 1),
                tokens=chunk_data.get('tokens', len(chunk_data['content']) // 4),
                start_char=0,  # Would need to track this
                end_char=len(chunk_data['content']),
                metadata=chunk_data
            )
            chunks.append(chunk)
        
        return chunks
    
    async def prepare_for_dataset_generation(
        self,
        collection: DocumentCollection,
        dataset_type: str
    ) -> pd.DataFrame:
        """
        Prepare document collection for dataset generation
        
        Args:
            collection: Processed document collection
            dataset_type: Type of dataset to generate
            
        Returns:
            DataFrame ready for dataset generation
        """
        # Get combined DataFrame
        df = collection.to_dataframe()
        
        # Add metadata columns
        df['dataset_type'] = dataset_type
        df['source_document_count'] = collection.total_documents
        
        # Apply any dataset-type specific transformations
        if dataset_type == 'qa_pairs' and collection.total_tokens > 100000:
            # For large QA datasets, might want to sample or prioritize
            print(f"[DocumentPipeline] Large dataset detected, optimizing for QA generation")
            # Could implement sampling strategies here
        
        return df

# Global instance
document_pipeline = DocumentPipeline()