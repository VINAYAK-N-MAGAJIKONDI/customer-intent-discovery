"""
Data Preprocessing Pipeline for Customer Intent Discovery Platform.

Handles dataset validation, profiling, text sanitization, normalization,
deduplication, and quality filtering while preserving semantic nuance.
"""

from typing import Dict, Any, Tuple, Optional
import re
import pandas as pd
import numpy as np


def clean_text(text: Any) -> str:
    """
    Sanitize and normalize raw customer query text while preserving
    contextual semantic information for transformer models.
    """
    if text is None or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Normalize unicode spaces and quotes
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')
    
    # Remove control characters and non-printable bytes
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    
    # Collapse multiple whitespaces / newlines
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def profile_dataset(df: pd.DataFrame, query_col: str) -> Dict[str, Any]:
    """
    Generate diagnostic profiling metrics for the uploaded dataset.
    """
    if query_col not in df.columns:
        raise ValueError(f"Column '{query_col}' not found in dataframe.")
    
    raw_series = df[query_col]
    total_rows = len(df)
    missing_count = int(raw_series.isna().sum())
    
    cleaned_series = raw_series.dropna().astype(str).map(clean_text)
    empty_after_clean = int((cleaned_series == "").sum())
    
    # Calculate lengths on non-empty queries
    valid_queries = cleaned_series[cleaned_series != ""]
    query_lengths = valid_queries.str.len()
    word_counts = valid_queries.str.split().map(len)
    
    duplicate_count = int(cleaned_series.duplicated().sum())
    
    return {
        "total_rows": total_rows,
        "missing_count": missing_count,
        "empty_after_clean": empty_after_clean,
        "duplicate_count": duplicate_count,
        "unique_queries": int(cleaned_series.nunique()),
        "avg_char_length": float(query_lengths.mean()) if not query_lengths.empty else 0.0,
        "median_char_length": float(query_lengths.median()) if not query_lengths.empty else 0.0,
        "min_char_length": int(query_lengths.min()) if not query_lengths.empty else 0,
        "max_char_length": int(query_lengths.max()) if not query_lengths.empty else 0,
        "avg_word_count": float(word_counts.mean()) if not word_counts.empty else 0.0,
    }


def preprocess_dataset(
    df: pd.DataFrame,
    query_col: str,
    drop_duplicates: bool = True,
    min_chars: int = 3,
    max_chars: int = 1500
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run complete preprocessing pipeline.
    
    Returns:
        processed_df: DataFrame with original columns + '_cleaned_query' and '_original_index'
        audit_report: Dictionary with count of records retained vs dropped and reasons.
    """
    if query_col not in df.columns:
        raise ValueError(f"Query column '{query_col}' is missing.")
        
    df_work = df.copy()
    df_work["_original_index"] = df_work.index
    total_original = len(df_work)

    if total_original == 0:
        audit_report = {
            "total_input_records": 0,
            "records_retained": 0,
            "total_dropped": 0,
            "dropped_missing": 0,
            "dropped_empty": 0,
            "dropped_too_short": 0,
            "dropped_too_long": 0,
            "dropped_duplicates": 0,
            "retention_rate_pct": 0.0
        }
        df_work["_cleaned_query"] = pd.Series(dtype=str)
        return df_work, audit_report
    
    # Step 1: Clean text
    df_work["_cleaned_query"] = df_work[query_col].map(clean_text).astype(str)
    
    # Step 2: Track drop reasons
    is_missing = df[query_col].isna()
    is_empty = df_work["_cleaned_query"] == ""
    is_too_short = df_work["_cleaned_query"].str.len() < min_chars
    is_too_long = df_work["_cleaned_query"].str.len() > max_chars
    
    drop_mask = is_missing | is_empty | is_too_short | is_too_long
    
    df_valid = df_work[~drop_mask].copy()
    
    dropped_missing = int(is_missing.sum())
    dropped_empty = int((~is_missing & is_empty).sum())
    dropped_too_short = int((~is_missing & ~is_empty & is_too_short).sum())
    dropped_too_long = int((~is_missing & ~is_empty & ~is_too_short & is_too_long).sum())
    dropped_duplicates = 0
    
    if drop_duplicates:
        initial_valid = len(df_valid)
        df_valid = df_valid.drop_duplicates(subset=["_cleaned_query"], keep="first").copy()
        dropped_duplicates = initial_valid - len(df_valid)
        
    audit_report = {
        "total_input_records": total_original,
        "records_retained": len(df_valid),
        "total_dropped": total_original - len(df_valid),
        "dropped_missing": dropped_missing,
        "dropped_empty": dropped_empty,
        "dropped_too_short": dropped_too_short,
        "dropped_too_long": dropped_too_long,
        "dropped_duplicates": dropped_duplicates,
        "retention_rate_pct": round(len(df_valid) / max(total_original, 1) * 100, 2)
    }
    
    df_valid = df_valid.reset_index(drop=True)
    return df_valid, audit_report
