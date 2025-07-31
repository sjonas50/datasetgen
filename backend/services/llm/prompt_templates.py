"""
Reusable prompt templates for LLM operations
"""

from typing import Dict, Any, List
import json


class PromptTemplates:
    """Collection of optimized prompt templates"""
    
    @staticmethod
    def schema_inference_extended() -> str:
        """Extended thinking prompt for complex schema inference"""
        return """You are a data engineering expert analyzing a dataset to infer its schema.

Please use extended thinking to:
1. Analyze the data structure and patterns
2. Identify data types beyond basic inference
3. Detect semantic relationships between fields
4. Recognize domain-specific formats (emails, phones, IDs, etc.)
5. Identify potential data quality issues
6. Suggest normalization opportunities

Provide a comprehensive schema analysis with:
- Field definitions with semantic meaning
- Detected relationships and dependencies
- Data quality observations
- Transformation recommendations

Think step-by-step about each field's purpose and relationships."""
    
    @staticmethod
    def data_quality_assessment() -> str:
        """Prompt for comprehensive data quality assessment"""
        return """Analyze this dataset's quality across multiple dimensions:

1. **Completeness**: Missing values, null patterns
2. **Consistency**: Format variations, naming conventions
3. **Accuracy**: Potential errors, outliers
4. **Validity**: Business rule violations, constraint issues
5. **Timeliness**: Date patterns, staleness indicators
6. **Uniqueness**: Duplicate detection, key violations

For each issue found:
- Describe the problem clearly
- Quantify the impact (rows/columns affected)
- Suggest remediation approaches
- Prioritize by severity (high/medium/low)

Provide an overall quality score (0-100) with justification."""
    
    @staticmethod
    def transform_code_generation() -> str:
        """Prompt for generating transformation code"""
        return """Generate production-ready code to transform data between schemas.

Requirements:
1. **Type Safety**: Handle all type conversions with validation
2. **Error Handling**: Graceful handling of edge cases
3. **Performance**: Optimize for large datasets
4. **Logging**: Add appropriate logging for debugging
5. **Documentation**: Clear comments and docstrings
6. **Testing**: Include example test cases

Code should be:
- Modular and reusable
- Following best practices for the language
- Handling null/missing values appropriately
- Validating business rules where applicable"""
    
    @staticmethod
    def pipeline_optimization() -> str:
        """Prompt for pipeline optimization analysis"""
        return """Analyze this data pipeline configuration and suggest optimizations.

Consider:
1. **Efficiency**: Identify bottlenecks and redundant operations
2. **Parallelization**: Steps that can run concurrently
3. **Caching**: Data that should be cached between steps
4. **Resource Usage**: Memory and compute optimization
5. **Error Recovery**: Checkpoint and retry strategies
6. **Cost**: Minimize LLM calls and data movement

Provide specific recommendations with:
- Performance impact estimates
- Implementation complexity
- Risk assessment
- Priority ranking"""
    
    @staticmethod
    def multi_modal_fusion() -> str:
        """Prompt for multi-modal data fusion strategy"""
        return """Design a fusion strategy for combining these different data modalities.

Analyze:
1. **Data Characteristics**: Format, frequency, volume of each source
2. **Temporal Alignment**: How to synchronize different time scales
3. **Semantic Relationships**: Meaningful connections between modalities
4. **Fusion Approach**: Early, late, or hybrid fusion recommendation
5. **Quality Preservation**: Maintaining signal quality during fusion
6. **Missing Data**: Handling incomplete modalities

Recommend:
- Optimal fusion architecture
- Preprocessing requirements
- Feature extraction methods
- Validation approaches"""
    
    @staticmethod
    def natural_language_to_pipeline() -> str:
        """Prompt for converting natural language to pipeline config"""
        return """Convert this natural language description into a structured data pipeline.

Steps:
1. **Parse Intent**: Understand the user's data processing goals
2. **Identify Sources**: Determine required data inputs
3. **Design Steps**: Break down into logical processing stages
4. **Select Tools**: Choose appropriate connectors and transformations
5. **Add Intelligence**: Identify where LLM processing adds value
6. **Include Validation**: Add quality checks and error handling

Generate a complete pipeline configuration with:
- Clear step definitions
- Appropriate error handling
- Resource requirements
- Expected outputs"""
    
    @staticmethod
    def document_table_extraction() -> str:
        """Prompt for extracting tables from documents"""
        return """Extract all tabular data from this document image.

For each table found:
1. **Structure**: Identify headers, rows, columns
2. **Content**: Extract all cell values accurately
3. **Format**: Preserve number formats, dates, currencies
4. **Context**: Note table title and surrounding context
5. **Quality**: Flag any unclear or ambiguous cells
6. **Metadata**: Page number, position, related text

Output as structured JSON with:
- Table array with proper structure
- Confidence scores for uncertain values
- Notes about data quality issues
- Suggested data types for each column"""
    
    @staticmethod
    def synthetic_data_generation() -> str:
        """Prompt for generating synthetic data"""
        return """Generate realistic synthetic data matching this schema and constraints.

Requirements:
1. **Statistical Properties**: Match distributions of original data
2. **Relationships**: Preserve correlations and dependencies
3. **Business Rules**: Respect all domain constraints
4. **Privacy**: Ensure no real data leakage
5. **Diversity**: Create varied, non-repetitive records
6. **Realism**: Use contextually appropriate values

Include:
- Realistic names, addresses, dates
- Proper format for IDs, codes, references
- Coherent relationships between fields
- Edge cases for testing"""
    
    @staticmethod
    def anomaly_explanation() -> str:
        """Prompt for explaining detected anomalies"""
        return """Explain these detected data anomalies in business terms.

For each anomaly:
1. **Description**: What makes this data point unusual
2. **Context**: How it differs from normal patterns
3. **Possible Causes**: Business reasons for the anomaly
4. **Impact**: Potential consequences if not addressed
5. **Recommendations**: How to investigate or fix

Provide explanations that are:
- Clear to non-technical users
- Actionable and specific
- Prioritized by importance
- Include next steps"""
    
    @classmethod
    def build_prompt(
        cls,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Build a complete prompt from template and context
        
        Args:
            template_name: Name of the template method
            context: Data to include in the prompt
            
        Returns:
            Complete prompt string
        """
        # Get base template
        template_method = getattr(cls, template_name, None)
        if not template_method:
            raise ValueError(f"Unknown template: {template_name}")
        
        base_template = template_method()
        
        # Add context
        context_str = json.dumps(context, indent=2) if isinstance(context, dict) else str(context)
        
        return f"{base_template}\n\nData Context:\n{context_str}"
    
    @classmethod
    def get_all_templates(cls) -> List[str]:
        """Get list of all available templates"""
        return [
            method for method in dir(cls)
            if not method.startswith('_') 
            and callable(getattr(cls, method))
            and method not in ['build_prompt', 'get_all_templates']
        ]