from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum

from services.llm.llm_factory import LLMFactory
from core.logging import get_logger

logger = get_logger(__name__)


class FusionStrategy(str, Enum):
    """Multi-modal fusion strategies"""
    EARLY = "early"      # Combine raw features before processing
    LATE = "late"        # Process separately, combine predictions
    HYBRID = "hybrid"    # Combination of early and late
    HIERARCHICAL = "hierarchical"  # Multi-level fusion
    ATTENTION = "attention"  # Attention-based weighting


class ModalityType(str, Enum):
    """Supported data modalities"""
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class FusionEngine:
    """
    MDF Framework-inspired multi-modal data fusion engine
    """
    
    def __init__(self, llm_provider: str = "claude"):
        self.llm_service = LLMFactory.create(llm_provider)
        self.fusion_strategies = {
            FusionStrategy.EARLY: self._early_fusion,
            FusionStrategy.LATE: self._late_fusion,
            FusionStrategy.HYBRID: self._hybrid_fusion,
            FusionStrategy.HIERARCHICAL: self._hierarchical_fusion,
            FusionStrategy.ATTENTION: self._attention_fusion
        }
    
    async def fuse_modalities(
        self,
        modalities: Dict[str, Any],
        strategy: FusionStrategy = FusionStrategy.HYBRID,
        target_schema: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Fuse multiple data modalities into a unified dataset
        
        Args:
            modalities: Dictionary of modality_name -> data
            strategy: Fusion strategy to use
            target_schema: Optional target schema for output
            
        Returns:
            Fused DataFrame
        """
        logger.info(f"Fusing {len(modalities)} modalities using {strategy} strategy")
        
        # Validate modalities
        validated_modalities = self._validate_modalities(modalities)
        
        # Apply fusion strategy
        fusion_func = self.fusion_strategies.get(strategy)
        if not fusion_func:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        
        fused_data = await fusion_func(validated_modalities)
        
        # Apply target schema if provided
        if target_schema:
            fused_data = await self._apply_schema(fused_data, target_schema)
        
        # Validate fusion quality
        quality_report = await self._assess_fusion_quality(fused_data, validated_modalities)
        logger.info(f"Fusion quality score: {quality_report.get('overall_score', 0):.2f}")
        
        return fused_data
    
    async def _early_fusion(
        self,
        modalities: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Early fusion: Combine raw features before processing
        Best for: Tightly coupled data with strong correlations
        """
        logger.info("Applying early fusion strategy")
        
        # Extract features from each modality
        feature_sets = []
        
        for name, modality in modalities.items():
            features = await self._extract_features(modality)
            # Prefix columns with modality name
            features.columns = [f"{name}_{col}" for col in features.columns]
            feature_sets.append(features)
        
        # Align and combine features
        if len(feature_sets) == 1:
            fused = feature_sets[0]
        else:
            # Use LLM to determine best alignment strategy
            alignment_strategy = await self._determine_alignment_strategy(feature_sets)
            fused = self._align_and_combine(feature_sets, alignment_strategy)
        
        # Apply dimensionality reduction if needed
        if fused.shape[1] > 100:
            fused = await self._reduce_dimensions(fused)
        
        return fused
    
    async def _late_fusion(
        self,
        modalities: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Late fusion: Process each modality independently, then combine
        Best for: Heterogeneous sources with different characteristics
        """
        logger.info("Applying late fusion strategy")
        
        processed_modalities = {}
        
        # Process each modality independently
        for name, modality in modalities.items():
            processed = await self._process_modality(modality)
            processed_modalities[name] = processed
        
        # Use LLM to determine combination logic
        combination_plan = await self._plan_combination(processed_modalities)
        
        # Combine processed modalities
        fused = self._execute_combination_plan(processed_modalities, combination_plan)
        
        return fused
    
    async def _hybrid_fusion(
        self,
        modalities: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Hybrid fusion: Strategic combination of early and late fusion
        Best for: Complex scenarios with mixed coupling
        """
        logger.info("Applying hybrid fusion strategy")
        
        # Analyze modality relationships
        relationships = await self._analyze_modality_relationships(modalities)
        
        # Group tightly coupled modalities for early fusion
        early_groups = relationships.get("tightly_coupled", [])
        late_modalities = relationships.get("loosely_coupled", [])
        
        # Apply early fusion to groups
        early_fused = {}
        for group in early_groups:
            group_data = {k: modalities[k] for k in group if k in modalities}
            if len(group_data) > 1:
                fused = await self._early_fusion(group_data)
                early_fused[f"group_{len(early_fused)}"] = {"data": fused, "type": ModalityType.TABULAR}
            else:
                # Single modality, just process
                name = list(group_data.keys())[0]
                early_fused[name] = group_data[name]
        
        # Add loosely coupled modalities
        for modality in late_modalities:
            if modality in modalities:
                early_fused[modality] = modalities[modality]
        
        # Apply late fusion to all
        return await self._late_fusion(early_fused)
    
    async def _hierarchical_fusion(
        self,
        modalities: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Hierarchical fusion: Multi-level fusion with intermediate representations
        Best for: Complex hierarchical relationships
        """
        logger.info("Applying hierarchical fusion strategy")
        
        # Build fusion hierarchy using LLM
        hierarchy = await self._build_fusion_hierarchy(modalities)
        
        # Execute fusion level by level
        current_level = modalities.copy()
        
        for level in hierarchy:
            next_level = {}
            
            for fusion_group in level:
                group_name = fusion_group["name"]
                group_modalities = {
                    k: current_level[k] 
                    for k in fusion_group["modalities"] 
                    if k in current_level
                }
                
                if len(group_modalities) > 1:
                    # Fuse this group
                    strategy = FusionStrategy(fusion_group.get("strategy", "early"))
                    fused = await self._apply_fusion_strategy(group_modalities, strategy)
                    next_level[group_name] = {"data": fused, "type": ModalityType.TABULAR}
                elif len(group_modalities) == 1:
                    # Pass through
                    next_level.update(group_modalities)
            
            # Add any unfused modalities
            for k, v in current_level.items():
                if not any(k in group["modalities"] for group in level):
                    next_level[k] = v
            
            current_level = next_level
            
            # Stop if we have a single result
            if len(current_level) == 1:
                break
        
        # Final fusion if needed
        if len(current_level) > 1:
            return await self._early_fusion(current_level)
        else:
            return list(current_level.values())[0]["data"]
    
    async def _attention_fusion(
        self,
        modalities: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Attention-based fusion: Dynamically weight modality contributions
        Best for: Scenarios where modality importance varies
        """
        logger.info("Applying attention-based fusion strategy")
        
        # Extract features from each modality
        features = {}
        for name, modality in modalities.items():
            features[name] = await self._extract_features(modality)
        
        # Calculate attention weights using LLM
        weights = await self._calculate_attention_weights(features)
        
        # Apply weighted fusion
        weighted_features = []
        for name, feature_df in features.items():
            weight = weights.get(name, 1.0)
            # Apply weight to numeric columns
            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
            feature_df[numeric_cols] = feature_df[numeric_cols] * weight
            weighted_features.append(feature_df)
        
        # Combine weighted features
        return self._align_and_combine(weighted_features, "outer")
    
    async def _extract_features(
        self,
        modality: Dict[str, Any]
    ) -> pd.DataFrame:
        """Extract features from a modality"""
        modality_type = modality.get("type", ModalityType.TABULAR)
        data = modality.get("data")
        
        if modality_type == ModalityType.TABULAR:
            if isinstance(data, pd.DataFrame):
                return data
            else:
                return pd.DataFrame(data)
        
        elif modality_type == ModalityType.TEXT:
            # Extract text features using LLM
            return await self._extract_text_features(data)
        
        elif modality_type == ModalityType.IMAGE:
            # Extract image features
            return await self._extract_image_features(data)
        
        elif modality_type == ModalityType.TEMPORAL:
            # Extract temporal features
            return await self._extract_temporal_features(data)
        
        else:
            # Generic feature extraction
            return pd.DataFrame({"feature": [str(data)]})
    
    async def _extract_text_features(self, text_data: Union[str, List[str]]) -> pd.DataFrame:
        """Extract features from text using LLM"""
        if isinstance(text_data, str):
            text_data = [text_data]
        
        prompt = """Extract structured features from these text samples:

1. Key entities (people, places, organizations)
2. Sentiment scores
3. Topics/themes
4. Numerical values mentioned
5. Dates and time references

Return as a table with one row per text sample."""
        
        result = await self.llm_service.complete(
            prompt + "\n\nTexts:\n" + "\n---\n".join(text_data[:10])  # Limit samples
        )
        
        # Parse result into DataFrame
        # This is simplified - real implementation would parse LLM output
        features = pd.DataFrame({
            "text_length": [len(t) for t in text_data],
            "word_count": [len(t.split()) for t in text_data]
        })
        
        return features
    
    async def _extract_image_features(self, image_data: Any) -> pd.DataFrame:
        """Extract features from images"""
        # Simplified - real implementation would extract visual features
        return pd.DataFrame({
            "image_feature": [1.0]
        })
    
    async def _extract_temporal_features(self, temporal_data: Any) -> pd.DataFrame:
        """Extract features from time series data"""
        if isinstance(temporal_data, pd.DataFrame):
            # Extract statistical features
            features = {
                "mean": temporal_data.mean(),
                "std": temporal_data.std(),
                "trend": self._calculate_trend(temporal_data)
            }
            return pd.DataFrame([features])
        
        return pd.DataFrame()
    
    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate simple trend coefficient"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Simple linear trend on first numeric column
            y = df[numeric_cols[0]].values
            x = np.arange(len(y))
            if len(x) > 1:
                return np.polyfit(x, y, 1)[0]
        return 0.0
    
    async def _determine_alignment_strategy(
        self,
        feature_sets: List[pd.DataFrame]
    ) -> str:
        """Determine how to align multiple feature sets"""
        # Analyze index compatibility
        index_types = [str(df.index.dtype) for df in feature_sets]
        
        if all(idx == index_types[0] for idx in index_types):
            # Same index type - try inner join first
            return "inner"
        else:
            # Different index types - use outer join
            return "outer"
    
    def _align_and_combine(
        self,
        feature_sets: List[pd.DataFrame],
        strategy: str = "outer"
    ) -> pd.DataFrame:
        """Align and combine multiple DataFrames"""
        if not feature_sets:
            return pd.DataFrame()
        
        if len(feature_sets) == 1:
            return feature_sets[0]
        
        # Start with first DataFrame
        result = feature_sets[0]
        
        # Join others based on strategy
        for df in feature_sets[1:]:
            if strategy == "inner":
                # Keep only matching indices
                result = result.join(df, how="inner", lsuffix="_left", rsuffix="_right")
            elif strategy == "outer":
                # Keep all indices
                result = result.join(df, how="outer", lsuffix="_left", rsuffix="_right")
            else:
                # Concatenate along axis
                result = pd.concat([result, df], axis=1)
        
        return result
    
    async def _reduce_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce dimensionality of features"""
        # Simplified - real implementation would use PCA/UMAP
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 50:
            # Keep top 50 columns by variance
            variances = df[numeric_cols].var()
            top_cols = variances.nlargest(50).index.tolist()
            
            # Keep non-numeric and top numeric columns
            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            return df[non_numeric + top_cols]
        
        return df
    
    def _validate_modalities(
        self,
        modalities: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Validate and standardize modality format"""
        validated = {}
        
        for name, modality in modalities.items():
            if isinstance(modality, dict) and "data" in modality:
                validated[name] = modality
            else:
                # Infer type and wrap
                if isinstance(modality, pd.DataFrame):
                    validated[name] = {
                        "data": modality,
                        "type": ModalityType.TABULAR
                    }
                elif isinstance(modality, (list, str)):
                    validated[name] = {
                        "data": modality,
                        "type": ModalityType.TEXT
                    }
                else:
                    validated[name] = {
                        "data": modality,
                        "type": ModalityType.TABULAR
                    }
        
        return validated
    
    async def _analyze_modality_relationships(
        self,
        modalities: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze relationships between modalities"""
        # Use LLM to analyze relationships
        modality_info = {
            name: {
                "type": m.get("type", "unknown"),
                "shape": m["data"].shape if hasattr(m["data"], "shape") else "N/A",
                "sample": str(m["data"])[:200] if not isinstance(m["data"], pd.DataFrame) else m["data"].head(2).to_dict()
            }
            for name, m in modalities.items()
        }
        
        prompt = f"""Analyze these data modalities and identify relationships:

{modality_info}

Classify modalities as:
1. Tightly coupled (should be fused early)
2. Loosely coupled (should be fused late)

Return as JSON with "tightly_coupled" and "loosely_coupled" lists."""
        
        result = await self.llm_service.complete(prompt)
        
        # Parse result
        import json
        try:
            relationships = json.loads(result.content)
            return relationships
        except:
            # Default grouping
            return {
                "tightly_coupled": [list(modalities.keys())],
                "loosely_coupled": []
            }
    
    async def _assess_fusion_quality(
        self,
        fused_data: pd.DataFrame,
        original_modalities: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess quality of fusion result"""
        quality_metrics = {
            "completeness": self._calculate_completeness(fused_data),
            "information_preservation": await self._calculate_information_preservation(
                fused_data, original_modalities
            ),
            "redundancy": self._calculate_redundancy(fused_data),
            "coherence": await self._calculate_coherence(fused_data)
        }
        
        # Overall score
        quality_metrics["overall_score"] = np.mean(list(quality_metrics.values()))
        
        return quality_metrics
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score"""
        total_cells = df.size
        non_null_cells = df.count().sum()
        
        return non_null_cells / total_cells if total_cells > 0 else 0.0
    
    async def _calculate_information_preservation(
        self,
        fused: pd.DataFrame,
        original: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate how much information was preserved"""
        # Simplified - count features
        original_features = sum(
            m["data"].shape[1] if hasattr(m["data"], "shape") else 1
            for m in original.values()
        )
        
        fused_features = fused.shape[1]
        
        return min(1.0, fused_features / original_features) if original_features > 0 else 1.0
    
    def _calculate_redundancy(self, df: pd.DataFrame) -> float:
        """Calculate redundancy score (lower is better)"""
        # Check correlation between numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs()
            # Get upper triangle
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Count high correlations
            high_corr = (upper_tri > 0.9).sum().sum()
            total_pairs = (numeric_df.shape[1] * (numeric_df.shape[1] - 1)) / 2
            
            redundancy = high_corr / total_pairs if total_pairs > 0 else 0
            
            # Return non-redundancy score
            return 1.0 - redundancy
        
        return 1.0
    
    async def _calculate_coherence(self, df: pd.DataFrame) -> float:
        """Calculate semantic coherence of fused data"""
        # Use LLM to assess coherence
        sample = df.head(5).to_dict()
        
        prompt = f"""Assess the coherence of this fused dataset:

{sample}

Rate from 0-1 how well the different data elements fit together and make semantic sense as a unified dataset."""
        
        # For now, return default score
        return 0.85