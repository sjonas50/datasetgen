"""
AI-Powered Dataset Generation Service
Transforms raw data into structured training datasets using Claude
"""

import json
from typing import Dict, Any, List, Optional
import pandas as pd
from services.claude_service import ClaudeService

class DatasetGenerator:
    """Generate training datasets from raw data using AI"""
    
    def __init__(self):
        self.claude_service = ClaudeService()
    
    async def generate_dataset(self, 
                             input_data: pd.DataFrame, 
                             dataset_type: str,
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a training dataset from input data
        
        Args:
            input_data: Raw input data
            dataset_type: Type of dataset to generate (e.g., 'qa_pairs', 'classification', 'ner', 'summarization')
            config: Configuration including prompts, examples, etc.
        
        Returns:
            Generated dataset with metadata
        """
        
        print(f"[DatasetGenerator] Starting generation for type: {dataset_type}")
        print(f"[DatasetGenerator] Input data shape: {input_data.shape}")
        print(f"[DatasetGenerator] Input columns: {list(input_data.columns)}")
        
        if input_data.empty:
            print(f"[DatasetGenerator] ERROR: Input DataFrame is empty!")
            return {
                "error": "No input data to generate dataset from",
                "success": False,
                "generated_df": pd.DataFrame()
            }
        
        if not self.claude_service.enabled:
            return {
                "error": "Claude service not available",
                "success": False,
                "generated_df": pd.DataFrame()
            }
        
        # Build the generation prompt based on dataset type
        if dataset_type == "qa_pairs":
            prompt = self._build_qa_generation_prompt(input_data, config)
        elif dataset_type == "classification":
            prompt = self._build_classification_prompt(input_data, config)
        elif dataset_type == "ner":
            prompt = self._build_ner_prompt(input_data, config)
        elif dataset_type == "summarization":
            prompt = self._build_summarization_prompt(input_data, config)
        elif dataset_type == "custom":
            # Handle custom dataset type with user instructions
            custom_instructions = config.get("custom_instructions", "")
            if custom_instructions:
                prompt = self._build_custom_prompt(input_data, config)
            else:
                prompt = config.get("custom_prompt", "")
        else:
            prompt = self._build_generic_prompt(input_data, config)
        
        # Generate the dataset using Claude
        try:
            response = await self.claude_service.client.messages.create(
                model=self.claude_service.model,
                max_tokens=4096,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse the response
            generated_content = response.content[0].text
            
            # Try to parse as JSON
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', generated_content)
                if json_match:
                    dataset_json = json.loads(json_match.group())
                    
                    # Convert to DataFrame
                    if isinstance(dataset_json, list):
                        generated_df = pd.DataFrame(dataset_json)
                    elif isinstance(dataset_json, dict) and 'data' in dataset_json:
                        generated_df = pd.DataFrame(dataset_json['data'])
                    else:
                        generated_df = pd.DataFrame([dataset_json])
                    
                    print(f"[DatasetGenerator] Successfully parsed JSON, generated {len(generated_df)} rows")
                    
                    return {
                        "success": True,
                        "generated_df": generated_df,
                        "dataset_type": dataset_type,
                        "row_count": len(generated_df),
                        "columns": list(generated_df.columns)
                    }
                else:
                    print(f"[DatasetGenerator] No JSON found, trying alternative parsing")
                    # Fallback: treat as CSV or structured text
                    lines = generated_content.strip().split('\n')
                    if len(lines) > 1 and ',' in lines[0]:
                        # Try to parse as CSV
                        import io
                        generated_df = pd.read_csv(io.StringIO(generated_content))
                    else:
                        # Try to parse based on dataset type
                        if dataset_type == "qa_pairs":
                            # Look for Q: and A: patterns
                            qa_pairs = []
                            for i in range(0, len(lines)-1, 2):
                                if 'Q:' in lines[i] or 'Question:' in lines[i]:
                                    question = lines[i].replace('Q:', '').replace('Question:', '').strip()
                                    if i+1 < len(lines):
                                        answer = lines[i+1].replace('A:', '').replace('Answer:', '').strip()
                                        if question and answer:
                                            qa_pairs.append({'question': question, 'answer': answer})
                            
                            if qa_pairs:
                                generated_df = pd.DataFrame(qa_pairs)
                            else:
                                generated_df = pd.DataFrame({"generated_text": [generated_content]})
                        else:
                            # Create single column DataFrame
                            generated_df = pd.DataFrame({"generated_text": [generated_content]})
                    
                    return {
                        "success": True,
                        "generated_df": generated_df,
                        "dataset_type": dataset_type,
                        "row_count": len(generated_df)
                    }
                    
            except Exception as parse_error:
                return {
                    "success": False,
                    "error": f"Failed to parse generated content: {str(parse_error)}",
                    "raw_content": generated_content,
                    "generated_df": pd.DataFrame()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generated_df": pd.DataFrame()
            }
    
    def _build_qa_generation_prompt(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """Build prompt for Q&A pair generation"""
        
        # Get sample content
        content_sample = self._get_content_sample(data)
        
        # Check if this is a scanned document placeholder
        is_scanned = "scanned PDF document" in content_sample or "Scanned/Image-based PDF" in content_sample
        
        # Count documents if we have source file info
        num_docs = len(data) if 'source_file' in data.columns else 1
        if num_docs > 1:
            print(f"[DatasetGenerator] Building Q&A prompt for {num_docs} documents")
        
        if is_scanned:
            prompt = f"""Generate a comprehensive question-answer training dataset based on this document description.

Document Information:
{content_sample}

Since this is a scanned document, create realistic Q&A pairs that would be typical for this type of professional document. Generate questions and answers as if you had access to the full document content.

IMPORTANT: You MUST generate actual question-answer pairs, NOT a summary of the document.

Requirements:
1. Create diverse Q&A pairs covering all potential aspects of such a document
2. Include questions about:
   - Document purpose and overview
   - Key sections and their content
   - Specific details that would appear in this type of document
   - Analysis and insights
   - Practical applications
3. Make answers detailed and realistic for the document type
4. Ensure variety in question types and complexity levels
5. Each entry MUST have a 'question' field with a complete question
6. Each entry MUST have an 'answer' field with a direct answer

Output format: Return ONLY a JSON array where each object has exactly two fields: 'question' and 'answer'.

Example:
[
  {{"question": "What is the purpose of this document?", "answer": "The purpose of this document is..."}},
  {{"question": "What are the main sections covered?", "answer": "The main sections include..."}}
]

Generate EXACTLY {config.get('min_examples', 20)} Q&A pairs. Do NOT generate a summary."""
        else:
            prompt = f"""Generate question-answer pairs for training a Q&A model based on the following content.

Content:
{content_sample}

IMPORTANT: You MUST generate actual question-answer pairs, NOT a summary of the content.

Requirements:
1. Generate diverse, high-quality question-answer pairs
2. Questions should cover different aspects of the content
3. Answers should be accurate and based only on the provided content
4. Include different types of questions (factual, analytical, inferential)
5. Each question must be a complete question that someone might ask about the content
6. Each answer must directly answer the corresponding question

{config.get('additional_instructions', '')}

Output format: You MUST return a JSON array where each object has exactly two fields: 'question' and 'answer'.

Example format:
[
  {{"question": "What is the main topic of the document?", "answer": "The main topic is [specific answer from content]"}},
  {{"question": "What are the key findings mentioned?", "answer": "The key findings are [specific findings from content]"}},
  {{"question": "How does the document describe [specific concept]?", "answer": "The document describes [concept] as [specific description from content]"}}
]

Generate EXACTLY {config.get('min_examples', 20)} Q&A pairs. Do NOT generate a summary or any other format."""
        
        return prompt
    
    def _build_classification_prompt(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """Build prompt for classification dataset generation"""
        
        content_sample = self._get_content_sample(data)
        
        # Check if this is a scanned document
        is_scanned = "scanned PDF document" in content_sample or "Scanned/Image-based PDF" in content_sample
        
        if is_scanned:
            # For scanned documents, use document-appropriate categories
            categories = config.get('categories', ['business', 'technical', 'legal', 'educational', 'other'])
            prompt = f"""Generate a text classification training dataset based on this document description.

Document Information:
{content_sample}

Categories: {', '.join(categories)}

Create multiple text snippets that would appear in this type of document, each with an appropriate classification label.

Requirements:
1. Generate {config.get('min_examples', 20)} different text examples
2. Each example should be 1-3 sentences that could appear in this document type
3. Ensure balanced representation across all categories
4. Make examples realistic for the document type

Output format: JSON array with 'text' and 'label' fields.
Example:
[
  {{"text": "The quarterly revenue increased by 15% compared to last year", "label": "business"}},
  {{"text": "The system architecture uses a microservices approach", "label": "technical"}}
]

Generate exactly {config.get('min_examples', 20)} classification examples."""
        else:
            categories = config.get('categories', ['positive', 'negative', 'neutral'])
            prompt = f"""Generate a classification training dataset based on the following content.

Content:
{content_sample}

Categories: {', '.join(categories)}

Requirements:
1. Create {config.get('min_examples', 20)} diverse text examples
2. Each example should be a distinct snippet from the content
3. Ensure balanced representation across categories
4. Examples should be realistic and relevant

Output format: JSON array with 'text' and 'label' fields.
Example:
[
  {{"text": "This is amazing!", "label": "positive"}},
  {{"text": "Not sure about this", "label": "neutral"}}
]

Generate exactly {config.get('min_examples', 20)} examples."""
        
        return prompt
    
    def _build_ner_prompt(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """Build prompt for NER dataset generation"""
        
        content_sample = self._get_content_sample(data)
        entity_types = config.get('entity_types', ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE'])
        
        # Check if this is a scanned document
        is_scanned = "scanned PDF document" in content_sample or "Scanned/Image-based PDF" in content_sample
        
        if is_scanned:
            prompt = f"""Generate a Named Entity Recognition (NER) training dataset based on this document description.

Document Information:
{content_sample}

Entity Types: {', '.join(entity_types)}

Create multiple sentences that would appear in this type of document, with entities labeled.

Requirements:
1. Generate {config.get('min_examples', 15)} different sentences
2. Each sentence should contain 1-4 entities
3. Include diverse entity types in realistic contexts
4. Calculate exact character positions for start/end

Output format: JSON array with 'text' and 'entities' fields.
Example:
[
  {{
    "text": "John Smith joined Microsoft on January 15, 2024",
    "entities": [
      {{"text": "John Smith", "label": "PERSON", "start": 0, "end": 10}},
      {{"text": "Microsoft", "label": "ORGANIZATION", "start": 18, "end": 27}},
      {{"text": "January 15, 2024", "label": "DATE", "start": 31, "end": 47}}
    ]
  }}
]

Generate exactly {config.get('min_examples', 15)} NER examples."""
        else:
            prompt = f"""Generate a Named Entity Recognition (NER) training dataset based on the following content.

Content:
{content_sample}

Entity Types: {', '.join(entity_types)}

Requirements:
1. Generate {config.get('min_examples', 15)} different sentences or passages
2. Extract or create sentences containing entities
3. Label all entities with exact character positions
4. Include diverse examples

Output format: JSON array with 'text' and 'entities' fields.
Example:
[
  {{
    "text": "John Smith works at OpenAI in San Francisco",
    "entities": [
      {{"text": "John Smith", "label": "PERSON", "start": 0, "end": 10}},
      {{"text": "OpenAI", "label": "ORGANIZATION", "start": 20, "end": 26}},
      {{"text": "San Francisco", "label": "LOCATION", "start": 30, "end": 43}}
    ]
  }}
]

Generate exactly {config.get('min_examples', 15)} examples."""
        
        return prompt
    
    def _build_summarization_prompt(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """Build prompt for summarization dataset generation"""
        
        content_sample = self._get_content_sample(data)
        
        # Check if this is a scanned document
        is_scanned = "scanned PDF document" in content_sample or "Scanned/Image-based PDF" in content_sample
        
        if is_scanned:
            prompt = f"""Generate a summarization training dataset based on this document description.

Document Information:
{content_sample}

Create multiple text-summary pairs that would be typical for this type of document.

Requirements:
1. Generate {config.get('min_examples', 15)} different text passages with summaries
2. Each 'text' should be a 3-5 paragraph section that could appear in this document
3. Each 'summary' should be 1-2 sentences capturing the key points
4. Make the content realistic for the document type
5. Vary the topics and complexity

Output format: JSON array with 'text' and 'summary' fields.
Example:
[
  {{
    "text": "The company's Q3 financial results showed significant growth across all sectors. Revenue increased by 23% year-over-year, reaching $2.4 billion. The strongest growth came from the cloud services division, which saw a 45% increase. Operating margins improved to 18%, up from 15% in the previous quarter.",
    "summary": "Q3 results showed 23% revenue growth to $2.4B, driven by 45% growth in cloud services."
  }}
]

Generate exactly {config.get('min_examples', 15)} text-summary pairs."""
        else:
            prompt = f"""Generate a summarization training dataset based on the following content.

Content:
{content_sample}

Requirements:
1. Create {config.get('min_examples', 15)} different text passages with summaries
2. Extract different sections from the content for each example
3. Each 'text' should be 3-5 sentences or a paragraph
4. Each 'summary' should be 1-2 sentences
5. Vary summary lengths and styles

Output format: JSON array with 'text' and 'summary' fields.
Example:
[
  {{
    "text": "Long passage about a topic that spans multiple sentences...",
    "summary": "Concise 1-2 sentence summary of the key points"
  }}
]

Generate exactly {config.get('min_examples', 15)} text-summary pairs."""
        
        return prompt
    
    def _build_custom_prompt(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """Build prompt for custom dataset generation based on user instructions"""
        
        content_sample = self._get_content_sample(data)
        custom_instructions = config.get('custom_instructions', '')
        
        prompt = f"""Generate a training dataset based on the following content and user requirements.

Content to analyze:
{content_sample}

User Requirements:
{custom_instructions}

Important:
1. Analyze the content and create a dataset that matches the user's requirements
2. Infer the appropriate format and structure based on the requirements
3. Generate diverse, high-quality examples
4. Output as a JSON array with appropriate fields

Generate at least {config.get('min_examples', 20)} examples."""
        
        return prompt
    
    def _build_generic_prompt(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """Build a generic dataset generation prompt"""
        
        content_sample = self._get_content_sample(data)
        
        prompt = f"""Generate a training dataset based on the following content and requirements.

Content:
{content_sample}

Dataset Requirements:
{config.get('requirements', 'Create a structured dataset suitable for machine learning')}

Output Format:
{config.get('output_format', 'JSON array with relevant fields')}

Additional Instructions:
{config.get('instructions', 'Generate diverse, high-quality examples')}

Generate at least {config.get('min_examples', 20)} examples."""
        
        return prompt
    
    def _get_content_sample(self, data: pd.DataFrame, max_chars: int = 3000) -> str:
        """Get a representative sample of content from the DataFrame"""
        
        print(f"[DatasetGenerator] Getting content sample from columns: {list(data.columns)}")
        print(f"[DatasetGenerator] DataFrame has {len(data)} rows")
        
        # Try to get text content
        text_columns = ['enhanced_content', 'text_content', 'content', 'text']
        
        all_documents = []
        source_files = []
        doc_metadata = []
        
        # Check if we have document metadata
        if 'document_id' in data.columns:
            # New document pipeline format
            unique_docs = data['document_id'].unique()
            print(f"[DatasetGenerator] Processing {len(unique_docs)} documents from new pipeline")
            
            for doc_id in unique_docs:
                doc_data = data[data['document_id'] == doc_id]
                
                # Get content for this document
                for col in text_columns:
                    if col in doc_data.columns:
                        content_parts = []
                        for _, row in doc_data.iterrows():
                            content = str(row[col]) if pd.notna(row[col]) else ""
                            if content.strip():
                                content_parts.append(content)
                        
                        if content_parts:
                            doc_content = '\n'.join(content_parts)
                            filename = doc_data.iloc[0].get('filename', f'Document {doc_id}')
                            doc_marker = f"\n\n=== {filename} ===\n\n"
                            all_documents.append(doc_marker + doc_content)
                            source_files.append(filename)
                            break
        
        # Check if we have source file information (legacy format)
        elif 'source_file' in data.columns or 'source_files' in data.columns:
            source_col = 'source_file' if 'source_file' in data.columns else 'source_files'
            source_files = data[source_col].dropna().unique().tolist()
            print(f"[DatasetGenerator] Processing {len(source_files)} source files (legacy format)")
        
        # If no documents collected yet, use legacy method
        if not all_documents:
            for col in text_columns:
                if col in data.columns:
                    print(f"[DatasetGenerator] Found text column: {col}")
                    # Process each row separately to maintain document boundaries
                    for idx, row in data.iterrows():
                        content = str(row[col]) if pd.notna(row[col]) else ""
                        if content.strip():
                            # Add document marker if we have multiple documents
                            if len(data) > 1 and ('source_file' in row or 'filename' in row):
                                doc_name = row.get('filename', row.get('source_file', f'Document {idx+1}'))
                                doc_marker = f"\n\n--- Document: {doc_name} ---\n\n"
                                all_documents.append(doc_marker + content)
                            else:
                                all_documents.append(content)
                    
                    if all_documents:
                        break
        
        if all_documents:
            # Join all documents
            all_text = '\n\n'.join(all_documents)
            print(f"[DatasetGenerator] Combined {len(all_documents)} documents, total length: {len(all_text)} characters")
            
            # Add summary if multiple documents
            if len(all_documents) > 1:
                summary = f"Processing {len(all_documents)} documents"
                if source_files:
                    summary += f": {', '.join(str(f) for f in source_files[:3])}"
                    if len(source_files) > 3:
                        summary += f" and {len(source_files) - 3} more"
                print(f"[DatasetGenerator] {summary}")
            
            return all_text[:max_chars] + ('...' if len(all_text) > max_chars else '')
        
        # Fallback: convert entire dataframe to string
        return str(data.to_dict('records'))[:max_chars]
    
    async def generate_dataset_chunked(
        self,
        chunks: List[Dict[str, Any]],
        dataset_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process multiple chunks and aggregate results
        
        Args:
            chunks: List of document chunks from DocumentProcessor
            dataset_type: Type of dataset to generate
            config: Configuration for generation
            
        Returns:
            Aggregated dataset results
        """
        print(f"[DatasetGenerator] Processing {len(chunks)} chunks for {dataset_type} dataset")
        
        all_results = []
        total_rows = 0
        failed_chunks = 0
        
        for i, chunk in enumerate(chunks):
            print(f"[DatasetGenerator] Processing chunk {i+1}/{len(chunks)}")
            
            try:
                # Create DataFrame from chunk
                chunk_df = pd.DataFrame([{
                    'content': chunk['content'],
                    'enhanced_content': chunk['content'],
                    'source_files': ', '.join(chunk['source_files']),
                    'chunk_info': f"Chunk {i+1}/{len(chunks)}"
                }])
                
                # Generate dataset for this chunk
                result = await self.generate_dataset(
                    chunk_df,
                    dataset_type,
                    config
                )
                
                if result.get('success') and result.get('row_count', 0) > 0:
                    all_results.append(result['generated_df'])
                    total_rows += result['row_count']
                else:
                    failed_chunks += 1
                    print(f"[DatasetGenerator] Chunk {i+1} failed or produced no rows")
                    
            except Exception as e:
                print(f"[DatasetGenerator] Error processing chunk {i+1}: {str(e)}")
                failed_chunks += 1
        
        # Aggregate results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Remove duplicates if any
            original_len = len(combined_df)
            combined_df = combined_df.drop_duplicates()
            if len(combined_df) < original_len:
                print(f"[DatasetGenerator] Removed {original_len - len(combined_df)} duplicate rows")
            
            return {
                "success": True,
                "generated_df": combined_df,
                "dataset_type": dataset_type,
                "row_count": len(combined_df),
                "chunks_processed": len(chunks) - failed_chunks,
                "chunks_failed": failed_chunks,
                "columns": list(combined_df.columns)
            }
        else:
            return {
                "success": False,
                "error": "No chunks produced results",
                "chunks_failed": failed_chunks,
                "generated_df": pd.DataFrame()
            }
    
    async def generate_dataset_streaming(
        self,
        input_data: pd.DataFrame,
        dataset_type: str,
        config: Dict[str, Any],
        callback: Optional[callable] = None
    ):
        """
        Generate dataset with streaming results
        
        Args:
            input_data: Input DataFrame
            dataset_type: Type of dataset
            config: Generation config
            callback: Optional callback for progress updates
            
        Yields:
            Generated rows as they're produced
        """
        # For now, implement basic streaming
        # In production, this could use SSE or WebSockets
        
        result = await self.generate_dataset(input_data, dataset_type, config)
        
        if result.get('success') and result.get('row_count', 0) > 0:
            df = result['generated_df']
            
            # Yield rows one by one
            for idx, row in df.iterrows():
                if callback:
                    callback({
                        'progress': (idx + 1) / len(df),
                        'current_row': idx + 1,
                        'total_rows': len(df)
                    })
                
                yield row.to_dict()

# Global instance
dataset_generator = DatasetGenerator()