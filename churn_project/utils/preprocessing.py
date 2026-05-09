"""
Preprocessing Utilities for Churn Prediction Pipeline
Handles validation, column checks, and data cleaning before pipeline inference.
"""

import logging
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Required columns for prediction (features only, no target)
REQUIRED_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Columns that should NOT be in prediction input
TARGET_COLUMN = "Churn"


def validate_uploaded_dataframe(
    df: pd.DataFrame,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that an uploaded DataFrame has required columns.

    Args:
        df: Uploaded customer dataframe.

    Returns:
        (is_valid, missing_columns, extra_columns)
    """
    uploaded_cols = set(df.columns.str.strip())
    required_cols = set(REQUIRED_COLUMNS)

    missing = sorted(required_cols - uploaded_cols)
    extra = sorted(uploaded_cols - required_cols - {TARGET_COLUMN})

    is_valid = len(missing) == 0
    return is_valid, missing, extra


def suggest_column_fixes(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-detect likely column name mismatches using fuzzy matching.
    Returns a mapping: {missing_col: best_match_in_df or None}.
    """
    from difflib import get_close_matches

    _, missing, _ = validate_uploaded_dataframe(df)
    suggestions = {}
    df_cols = list(df.columns)

    for col in missing:
        matches = get_close_matches(col.lower(), [c.lower() for c in df_cols], n=1, cutoff=0.6)
        if matches:
            # Map back to original casing
            matched_lower = matches[0]
            original = next((c for c in df_cols if c.lower() == matched_lower), None)
            suggestions[col] = original
        else:
            suggestions[col] = None

    return suggestions


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning before passing to pipeline:
    - Strip whitespace from string columns
    - Convert TotalCharges to numeric (may contain spaces in raw data)
    - Fill missing numerical values with median, categorical with mode
    """
    df = df.copy()

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # TotalCharges sometimes has blank strings in raw telecom data
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Strip whitespace from object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Fill NaN: numerical → median, categorical → mode
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled NaN in '{col}' with median={median_val:.2f}")

    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Filled NaN in '{col}' with mode='{mode_val}'")

    return df


def prepare_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the required feature columns in correct order.
    Drops the target column if present.
    """
    df = df.copy()
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])
    return df[REQUIRED_COLUMNS]


def get_required_columns() -> List[str]:
    return REQUIRED_COLUMNS.copy()
