"""
Individual data processing tasks
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats
import asyncio

from services.claude_service import ClaudeService

# PII detection patterns
PII_PATTERNS = {
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b',
    'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b',
    'ip_address': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
}

def quality_validation_task(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data quality
    """
    results = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "issues": [],
        "column_analysis": {},
        "quality_score": 100.0
    }
    
    total_cells = df.shape[0] * df.shape[1]
    issue_cells = 0
    
    # Check for missing values
    missing_by_column = df.isnull().sum()
    for col, missing_count in missing_by_column.items():
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            severity = "critical" if missing_pct > 50 else "high" if missing_pct > 20 else "medium"
            
            results["issues"].append({
                "type": "missing_values",
                "column": col,
                "count": int(missing_count),
                "percentage": round(missing_pct, 2),
                "severity": severity
            })
            issue_cells += missing_count
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        results["issues"].append({
            "type": "duplicate_rows",
            "count": int(duplicate_count),
            "percentage": round((duplicate_count / len(df)) * 100, 2),
            "severity": "medium"
        })
        issue_cells += duplicate_count * df.shape[1]
    
    # Column-level analysis
    for col in df.columns:
        col_data = df[col]
        col_analysis = {
            "dtype": str(col_data.dtype),
            "unique_values": int(col_data.nunique()),
            "missing_count": int(col_data.isnull().sum()),
            "missing_percentage": round((col_data.isnull().sum() / len(df)) * 100, 2)
        }
        
        # Numeric column statistics
        if pd.api.types.is_numeric_dtype(col_data):
            col_analysis.update({
                "mean": float(col_data.mean()) if not col_data.empty else None,
                "std": float(col_data.std()) if not col_data.empty else None,
                "min": float(col_data.min()) if not col_data.empty else None,
                "max": float(col_data.max()) if not col_data.empty else None,
                "q25": float(col_data.quantile(0.25)) if not col_data.empty else None,
                "q50": float(col_data.quantile(0.50)) if not col_data.empty else None,
                "q75": float(col_data.quantile(0.75)) if not col_data.empty else None,
            })
            
            # Check for outliers using IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                col_analysis["outlier_count"] = int(outliers)
                results["issues"].append({
                    "type": "outliers",
                    "column": col,
                    "count": int(outliers),
                    "severity": "low"
                })
        
        # String column analysis
        elif pd.api.types.is_string_dtype(col_data):
            col_analysis.update({
                "avg_length": col_data.str.len().mean() if not col_data.empty else None,
                "max_length": col_data.str.len().max() if not col_data.empty else None,
                "empty_strings": int((col_data == "").sum()),
                "whitespace_issues": int(col_data.str.strip().ne(col_data).sum())
            })
        
        results["column_analysis"][col] = col_analysis
    
    # Calculate quality score
    if total_cells > 0:
        results["quality_score"] = max(0, 100 - (issue_cells / total_cells) * 100)
    
    # Check against minimum quality score if specified
    if config.get("min_quality_score") and results["quality_score"] < config["min_quality_score"]:
        if config.get("fail_on_critical"):
            raise ValueError(f"Quality score {results['quality_score']:.1f} below minimum {config['min_quality_score']}")
    
    return results

def pii_detection_task(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect and handle PII data
    """
    pii_found = {}
    processed_df = df.copy()
    action = config.get("action", "mask")
    patterns_to_check = config.get("patterns", list(PII_PATTERNS.keys()))
    
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        
        col_pii = {}
        for pii_type in patterns_to_check:
            if pii_type not in PII_PATTERNS:
                continue
            
            pattern = PII_PATTERNS[pii_type]
            # Convert to string and check for matches
            matches = df[col].astype(str).str.contains(pattern, regex=True, na=False)
            match_count = matches.sum()
            
            if match_count > 0:
                col_pii[pii_type] = int(match_count)
                
                # Apply action
                if action == "mask":
                    if pii_type == "ssn":
                        processed_df.loc[matches, col] = df.loc[matches, col].str.replace(
                            pattern, "XXX-XX-XXXX", regex=True
                        )
                    elif pii_type == "email":
                        processed_df.loc[matches, col] = df.loc[matches, col].str.replace(
                            pattern, "****@****.***", regex=True
                        )
                    elif pii_type == "phone":
                        processed_df.loc[matches, col] = df.loc[matches, col].str.replace(
                            pattern, "XXX-XXX-XXXX", regex=True
                        )
                    elif pii_type == "credit_card":
                        processed_df.loc[matches, col] = df.loc[matches, col].str.replace(
                            pattern, "XXXX-XXXX-XXXX-XXXX", regex=True
                        )
                elif action == "remove":
                    processed_df.loc[matches, col] = ""
        
        if col_pii:
            pii_found[col] = col_pii
    
    return {
        "processed_df": processed_df,
        "report": {
            "pii_found": len(pii_found) > 0,
            "columns_with_pii": list(pii_found.keys()),
            "pii_details": pii_found,
            "action_taken": action,
            "total_pii_instances": sum(sum(col.values()) for col in pii_found.values())
        }
    }

def data_cleaning_task(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and standardize data
    """
    cleaned_df = df.copy()
    operations_performed = []
    
    # Remove duplicates
    if config.get("remove_duplicates", True):
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        if len(cleaned_df) < initial_rows:
            operations_performed.append({
                "operation": "remove_duplicates",
                "rows_removed": initial_rows - len(cleaned_df)
            })
    
    # Trim whitespace
    if config.get("trim_whitespace", True):
        string_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in string_columns:
            # Only apply strip to non-null string values
            mask = cleaned_df[col].notna()
            cleaned_df.loc[mask, col] = cleaned_df.loc[mask, col].astype(str).str.strip()
        if len(string_columns) > 0:
            operations_performed.append({
                "operation": "trim_whitespace",
                "columns_affected": list(string_columns)
            })
    
    # Handle missing values
    missing_strategy = config.get("handle_missing", "drop")
    if missing_strategy == "drop":
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        if len(cleaned_df) < initial_rows:
            operations_performed.append({
                "operation": "drop_missing",
                "rows_removed": initial_rows - len(cleaned_df)
            })
    elif missing_strategy == "fill":
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                else:
                    cleaned_df[col].fillna("", inplace=True)
        operations_performed.append({
            "operation": "fill_missing",
            "strategy": "mean for numeric, empty string for text"
        })
    
    # Standardize casing
    if config.get("standardize_casing", True):
        string_columns = cleaned_df.select_dtypes(include=['object']).columns
        casing = config.get("casing_type", "lower")
        for col in string_columns:
            if casing == "lower":
                cleaned_df[col] = cleaned_df[col].str.lower()
            elif casing == "upper":
                cleaned_df[col] = cleaned_df[col].str.upper()
            elif casing == "title":
                cleaned_df[col] = cleaned_df[col].str.title()
        
        if len(string_columns) > 0:
            operations_performed.append({
                "operation": "standardize_casing",
                "casing_type": casing,
                "columns_affected": list(string_columns)
            })
    
    # Remove invalid values
    if config.get("remove_invalid", False):
        # Remove rows with invalid numeric values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            cleaned_df = cleaned_df[~cleaned_df[col].isin([np.inf, -np.inf])]
    
    return {
        "cleaned_df": cleaned_df,
        "operations": operations_performed,
        "initial_rows": len(df),
        "final_rows": len(cleaned_df),
        "rows_modified": len(df) - len(cleaned_df)
    }

def outlier_detection_task(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect and handle outliers
    """
    processed_df = df.copy()
    outliers_found = {}
    method = config.get("method", "zscore")
    threshold = config.get("threshold", 3)
    action = config.get("action", "flag")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        col_data = df[col].dropna()
        
        if len(col_data) < 3:  # Not enough data for outlier detection
            continue
        
        if method == "zscore":
            z_scores = np.abs(stats.zscore(col_data))
            outlier_mask = z_scores > threshold
        elif method == "iqr":
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (col_data < (Q1 - threshold * IQR)) | (col_data > (Q3 + threshold * IQR))
        else:
            continue
        
        outlier_indices = col_data[outlier_mask].index
        
        if len(outlier_indices) > 0:
            outliers_found[col] = {
                "count": len(outlier_indices),
                "indices": outlier_indices.tolist(),
                "values": col_data[outlier_mask].tolist()
            }
            
            if action == "remove":
                processed_df = processed_df.drop(outlier_indices)
            elif action == "cap":
                if method == "iqr":
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    processed_df.loc[processed_df[col] < lower_bound, col] = lower_bound
                    processed_df.loc[processed_df[col] > upper_bound, col] = upper_bound
            elif action == "flag":
                # Add a flag column
                flag_col = f"{col}_outlier"
                processed_df[flag_col] = False
                processed_df.loc[outlier_indices, flag_col] = True
    
    return {
        "processed_df": processed_df,
        "outliers": outliers_found,
        "method": method,
        "threshold": threshold,
        "action": action,
        "total_outliers": sum(info["count"] for info in outliers_found.values())
    }

async def data_transformation_task(data: List[Dict], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform data using Claude AI
    """
    df = pd.DataFrame(data)
    transformation_request = config.get("prompt_template", "Transform the data")
    target_schema = config.get("target_schema")
    
    # Use Claude for transformation
    claude_service = ClaudeService()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        transformed_df = loop.run_until_complete(
            claude_service.transform_data(df, transformation_request, target_schema)
        )
        
        return {
            "transformed_data": transformed_df.to_dict(orient='records'),
            "transformation_log": {
                "input_rows": len(df),
                "output_rows": len(transformed_df),
                "input_columns": list(df.columns),
                "output_columns": list(transformed_df.columns),
                "request": transformation_request
            }
        }
    except Exception as e:
        return {
            "transformed_data": data,  # Return original data on error
            "transformation_log": {
                "error": str(e),
                "request": transformation_request
            }
        }

def schema_validation_task(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data against a schema
    """
    schema = config.get("schema", {})
    strict_mode = config.get("strict_mode", True)
    allow_extra_columns = config.get("allow_extra_columns", False)
    
    validation_errors = []
    warnings = []
    
    # Check required columns
    required_columns = schema.get("required_columns", [])
    for col in required_columns:
        if col not in df.columns:
            validation_errors.append({
                "type": "missing_column",
                "column": col,
                "message": f"Required column '{col}' is missing"
            })
    
    # Check extra columns
    if not allow_extra_columns and "columns" in schema:
        expected_columns = set(schema["columns"])
        actual_columns = set(df.columns)
        extra_columns = actual_columns - expected_columns
        if extra_columns:
            message = f"Extra columns found: {', '.join(extra_columns)}"
            if strict_mode:
                validation_errors.append({
                    "type": "extra_columns",
                    "columns": list(extra_columns),
                    "message": message
                })
            else:
                warnings.append(message)
    
    # Check data types
    expected_types = schema.get("column_types", {})
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            
            # Type mapping
            type_compatible = False
            if expected_type in ["int", "integer"] and "int" in actual_type:
                type_compatible = True
            elif expected_type in ["float", "decimal", "numeric"] and "float" in actual_type:
                type_compatible = True
            elif expected_type in ["str", "string", "text"] and "object" in actual_type:
                type_compatible = True
            elif expected_type in ["bool", "boolean"] and "bool" in actual_type:
                type_compatible = True
            elif expected_type in ["date", "datetime"] and "datetime" in actual_type:
                type_compatible = True
            
            if not type_compatible:
                message = f"Column '{col}' has type '{actual_type}' but expected '{expected_type}'"
                if strict_mode:
                    validation_errors.append({
                        "type": "type_mismatch",
                        "column": col,
                        "expected": expected_type,
                        "actual": actual_type,
                        "message": message
                    })
                else:
                    warnings.append(message)
    
    # Check constraints
    constraints = schema.get("constraints", {})
    for col, col_constraints in constraints.items():
        if col not in df.columns:
            continue
        
        # Unique constraint
        if col_constraints.get("unique", False):
            duplicate_count = df[col].duplicated().sum()
            if duplicate_count > 0:
                validation_errors.append({
                    "type": "unique_violation",
                    "column": col,
                    "duplicate_count": int(duplicate_count),
                    "message": f"Column '{col}' has {duplicate_count} duplicate values"
                })
        
        # Not null constraint
        if col_constraints.get("not_null", False):
            null_count = df[col].isnull().sum()
            if null_count > 0:
                validation_errors.append({
                    "type": "null_violation",
                    "column": col,
                    "null_count": int(null_count),
                    "message": f"Column '{col}' has {null_count} null values"
                })
        
        # Min/max constraints for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            if "min" in col_constraints:
                min_val = col_constraints["min"]
                violations = (df[col] < min_val).sum()
                if violations > 0:
                    validation_errors.append({
                        "type": "min_violation",
                        "column": col,
                        "min_value": min_val,
                        "violation_count": int(violations),
                        "message": f"Column '{col}' has {violations} values below minimum {min_val}"
                    })
            
            if "max" in col_constraints:
                max_val = col_constraints["max"]
                violations = (df[col] > max_val).sum()
                if violations > 0:
                    validation_errors.append({
                        "type": "max_violation",
                        "column": col,
                        "max_value": max_val,
                        "violation_count": int(violations),
                        "message": f"Column '{col}' has {violations} values above maximum {max_val}"
                    })
    
    return {
        "valid": len(validation_errors) == 0,
        "errors": validation_errors,
        "warnings": warnings,
        "row_count": len(df),
        "column_count": len(df.columns)
    }