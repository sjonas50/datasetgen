"""
Enhanced Dataset Generator with Batch Processing
Handles large-scale dataset generation without timeouts
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import math
from services.claude_service import ClaudeService

class DatasetGeneratorV2:
    """Generate large training datasets with batch processing"""
    
    def __init__(self):
        self.claude_service = ClaudeService()
        self.batch_size = 100  # Generate 100 examples per batch
        self.max_concurrent_batches = 3  # Process 3 batches concurrently
        self.timeout_per_batch = 30  # 30 seconds per batch
        
    async def generate_dataset_batch(self, 
                                   content_chunk: str,
                                   dataset_type: str,
                                   batch_num: int,
                                   total_batches: int,
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single batch of dataset examples"""
        
        print(f"[DatasetGeneratorV2] Generating batch {batch_num}/{total_batches} for {dataset_type}")
        
        # Build batch-specific prompt
        prompt = self._build_batch_prompt(
            content_chunk, 
            dataset_type, 
            batch_num, 
            config
        )
        
        try:
            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                self.claude_service.client.messages.create(
                    model=self.claude_service.model,
                    max_tokens=self.claude_service.max_tokens,
                    temperature=0.3,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                ),
                timeout=self.timeout_per_batch
            )
            
            # Parse response
            generated_content = response.content[0].text
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', generated_content)
            if json_match:
                dataset_json = json.loads(json_match.group())
                
                if isinstance(dataset_json, list):
                    df = pd.DataFrame(dataset_json)
                elif isinstance(dataset_json, dict) and 'data' in dataset_json:
                    df = pd.DataFrame(dataset_json['data'])
                else:
                    df = pd.DataFrame([dataset_json])
                
                print(f"[DatasetGeneratorV2] Batch {batch_num} generated {len(df)} rows")
                
                return {
                    "success": True,
                    "batch_df": df,
                    "batch_num": batch_num,
                    "row_count": len(df)
                }
            else:
                return {
                    "success": False,
                    "batch_num": batch_num,
                    "error": "No JSON found in response"
                }
                
        except asyncio.TimeoutError:
            print(f"[DatasetGeneratorV2] Batch {batch_num} timed out")
            return {
                "success": False,
                "batch_num": batch_num,
                "error": "Batch generation timed out"
            }
        except Exception as e:
            print(f"[DatasetGeneratorV2] Batch {batch_num} error: {str(e)}")
            return {
                "success": False,
                "batch_num": batch_num,
                "error": str(e)
            }
    
    async def generate_dataset_parallel(self,
                                      input_data: pd.DataFrame,
                                      dataset_type: str,
                                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate large dataset using parallel batch processing"""
        
        print(f"[DatasetGeneratorV2] Starting parallel generation for {dataset_type}")
        
        # Get content
        content = self._get_combined_content(input_data)
        content_length = len(content)
        
        # Determine total examples needed
        target_examples = config.get('target_rows', 1000)  # Default to 1000
        examples_per_batch = min(self.batch_size, 100)
        total_batches = math.ceil(target_examples / examples_per_batch)
        
        print(f"[DatasetGeneratorV2] Target: {target_examples} examples in {total_batches} batches")
        
        # Split content into chunks for each batch
        chunk_size = max(1000, content_length // max(total_batches, 1))
        content_chunks = []
        
        if content_length > chunk_size * 2:
            # For large content, use overlapping chunks
            for i in range(total_batches):
                start = min(i * chunk_size // 2, content_length - chunk_size)
                end = min(start + chunk_size, content_length)
                content_chunks.append(content[start:end])
        else:
            # For smaller content, reuse the same content
            content_chunks = [content] * total_batches
        
        # Process batches with concurrency limit
        all_results = []
        failed_batches = []
        
        # Process in groups to limit concurrency
        for i in range(0, total_batches, self.max_concurrent_batches):
            batch_group = []
            
            # Create tasks for this group
            for j in range(i, min(i + self.max_concurrent_batches, total_batches)):
                task = self.generate_dataset_batch(
                    content_chunks[j],
                    dataset_type,
                    j + 1,
                    total_batches,
                    {**config, 'examples_per_batch': examples_per_batch}
                )
                batch_group.append(task)
            
            # Execute batch group
            group_results = await asyncio.gather(*batch_group, return_exceptions=True)
            
            # Process results
            for result in group_results:
                if isinstance(result, Exception):
                    failed_batches.append({"error": str(result)})
                elif isinstance(result, dict):
                    if result.get('success') and 'batch_df' in result:
                        all_results.append(result['batch_df'])
                    else:
                        failed_batches.append(result)
            
            # Add small delay between groups to avoid rate limiting
            if i + self.max_concurrent_batches < total_batches:
                await asyncio.sleep(1)
        
        # Combine all successful batches
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Remove duplicates
            original_len = len(combined_df)
            combined_df = combined_df.drop_duplicates()
            if len(combined_df) < original_len:
                print(f"[DatasetGeneratorV2] Removed {original_len - len(combined_df)} duplicates")
            
            # Trim to target size if we generated too many
            if len(combined_df) > target_examples:
                combined_df = combined_df.head(target_examples)
            
            return {
                "success": True,
                "generated_df": combined_df,
                "dataset_type": dataset_type,
                "row_count": len(combined_df),
                "batches_completed": len(all_results),
                "batches_failed": len(failed_batches),
                "columns": list(combined_df.columns)
            }
        else:
            return {
                "success": False,
                "error": "All batches failed",
                "failed_batches": failed_batches,
                "generated_df": pd.DataFrame()
            }
    
    def _build_batch_prompt(self, content: str, dataset_type: str, batch_num: int, config: Dict[str, Any]) -> str:
        """Build prompt for a specific batch"""
        
        examples_per_batch = config.get('examples_per_batch', 100)
        
        if dataset_type == "qa_pairs":
            prompt = f"""Generate question-answer pairs for training a Q&A model based on this content.

Content:
{content[:3000]}{"..." if len(content) > 3000 else ""}

This is batch {batch_num}. Generate EXACTLY {examples_per_batch} UNIQUE question-answer pairs.

Requirements:
1. Questions must be diverse and cover different aspects
2. Include factual, analytical, inferential, and application questions
3. Vary question complexity and length
4. Answers must be accurate and based on the content
5. Make questions specific and detailed
6. DO NOT repeat questions from previous batches

Output format: JSON array with 'question' and 'answer' fields.

Generate EXACTLY {examples_per_batch} unique Q&A pairs for this batch."""

        elif dataset_type == "classification":
            categories = config.get('categories', ['positive', 'negative', 'neutral'])
            prompt = f"""Generate classification training data.

Content:
{content[:3000]}{"..." if len(content) > 3000 else ""}

This is batch {batch_num}. Generate EXACTLY {examples_per_batch} text samples with labels.

Categories: {', '.join(categories)}

Requirements:
1. Extract diverse text snippets
2. Ensure balanced distribution across categories
3. Vary text length (1-3 sentences)
4. Make each example unique

Output format: JSON array with 'text' and 'label' fields.

Generate EXACTLY {examples_per_batch} classification examples."""

        elif dataset_type == "ner":
            entity_types = config.get('entity_types', ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE'])
            prompt = f"""Generate NER training data.

Content:
{content[:3000]}{"..." if len(content) > 3000 else ""}

Batch {batch_num}: Generate EXACTLY {examples_per_batch} sentences with entities.

Entity Types: {', '.join(entity_types)}

Requirements:
1. Each sentence must contain 1-4 entities
2. Calculate exact character positions
3. Use diverse sentence structures
4. Include all entity types

Output format: JSON array with 'text' and 'entities' fields.
Each entity must have: text, label, start, end

Generate EXACTLY {examples_per_batch} NER examples."""

        elif dataset_type == "summarization":
            prompt = f"""Generate summarization training data.

Content:
{content[:3000]}{"..." if len(content) > 3000 else ""}

Batch {batch_num}: Generate EXACTLY {examples_per_batch} text-summary pairs.

Requirements:
1. Each 'text' should be 3-5 sentences
2. Each 'summary' should be 1-2 sentences
3. Extract different sections for variety
4. Make summaries concise but informative

Output format: JSON array with 'text' and 'summary' fields.

Generate EXACTLY {examples_per_batch} summarization pairs."""

        else:
            # Custom dataset type
            prompt = f"""Generate training data based on user requirements.

Content:
{content[:3000]}{"..." if len(content) > 3000 else ""}

Batch {batch_num}: Generate EXACTLY {examples_per_batch} examples.

{config.get('custom_instructions', 'Generate diverse training examples')}

Output as JSON array. Generate EXACTLY {examples_per_batch} examples."""
        
        return prompt
    
    def _get_combined_content(self, data: pd.DataFrame) -> str:
        """Get all content from dataframe"""
        text_columns = ['enhanced_content', 'text_content', 'content', 'text']
        
        all_content = []
        for col in text_columns:
            if col in data.columns:
                for _, row in data.iterrows():
                    content = str(row[col]) if pd.notna(row[col]) else ""
                    if content.strip() and len(content) > 50:
                        all_content.append(content)
        
        return '\n\n'.join(all_content)
    
    async def generate_dataset_streaming(self,
                                       input_data: pd.DataFrame,
                                       dataset_type: str,
                                       config: Dict[str, Any],
                                       progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Generate dataset with streaming progress updates"""
        
        target_examples = config.get('target_rows', 1000)
        examples_per_batch = min(self.batch_size, 100)
        total_batches = math.ceil(target_examples / examples_per_batch)
        
        all_results = []
        total_generated = 0
        
        # Get content
        content = self._get_combined_content(input_data)
        
        for batch_num in range(1, total_batches + 1):
            # Generate single batch
            result = await self.generate_dataset_batch(
                content,
                dataset_type,
                batch_num,
                total_batches,
                {**config, 'examples_per_batch': examples_per_batch}
            )
            
            if result.get('success') and 'batch_df' in result:
                all_results.append(result['batch_df'])
                total_generated += len(result['batch_df'])
                
                # Update progress
                if progress_callback:
                    progress_callback({
                        'batch': batch_num,
                        'total_batches': total_batches,
                        'rows_generated': total_generated,
                        'target_rows': target_examples,
                        'percentage': (batch_num / total_batches) * 100
                    })
                
                # Stop if we have enough
                if total_generated >= target_examples:
                    break
            
            # Small delay between batches
            if batch_num < total_batches:
                await asyncio.sleep(0.5)
        
        # Combine results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df = combined_df.drop_duplicates()
            
            if len(combined_df) > target_examples:
                combined_df = combined_df.head(target_examples)
            
            return {
                "success": True,
                "generated_df": combined_df,
                "dataset_type": dataset_type,
                "row_count": len(combined_df),
                "columns": list(combined_df.columns)
            }
        else:
            return {
                "success": False,
                "error": "No batches succeeded",
                "generated_df": pd.DataFrame()
            }