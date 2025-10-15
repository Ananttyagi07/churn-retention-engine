"""
Data Cleaning and Preprocessing Utilities

This module provides well-tested functions to:
- Handle missing data with imputation strategies
- Cap (winsorize) outliers using the IQR (Interquartile Range) rule or percentile limits
- Normalize and standardize categorical values (labels consolidation)
- Deduplicate rows
- Designed for structured tabular data (Pandas DataFrames) in ML workflows

Best practices based on latest guides and examples[web:190][web:191][web:193][web:195][web:200][web:192][web:194].
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Optional, Dict, Any, Union


def handle_missing(
    df: pd.DataFrame,
    strategy: str = "auto",
    fill_numeric: float = 0.0,
    fill_categorical: str = "missing",
    drop_threshold: float = 0.85,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    - For mostly-missing columns (over drop_threshold), drop them.
    - For numeric columns, fill with median (if strategy=auto), or fixed value.
    - For categorical, fill with mode or fixed string.

    Args:
        df: Input dataframe.
        strategy: "auto", "median", "mean", "mode", or "constant".
        fill_numeric: Used if strategy="constant" for numeric columns.
        fill_categorical: Used if strategy="constant" for categoricals.
        drop_threshold: Drop columns where non-null ratio < drop_threshold.
        inplace: If True, mutate df in place.

    Returns:
        Cleaned dataframe with missing values handled.
    """
    if not inplace:
        df = df.copy()

    # Drop columns with excess missingness
    null_ratios = df.isnull().mean()
    cols_to_drop = null_ratios[null_ratios > (1 - drop_threshold)].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping mostly-missing columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # Impute numerics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if strategy == "auto" or strategy == "median":
            val = df[col].median()
        elif strategy == "mean":
            val = df[col].mean()
        elif strategy == "mode":
            val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
        elif strategy == "constant":
            val = fill_numeric
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")
        if df[col].isnull().any():
            logger.info(f"Imputing missing values in '{col}' with: {val}")
            df[col].fillna(val, inplace=True)

    # Impute categoricals
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in categorical_cols:
        if strategy in ["auto", "mode"]:
            val = df[col].mode().iloc[0] if not df[col].mode().empty else fill_categorical
        elif strategy == "constant":
            val = fill_categorical
        else:
            raise ValueError(f"Unknown missing data strategy for categoricals: {strategy}")
        if df[col].isnull().any():
            logger.info(f"Imputing missing values in '{col}' with: '{val}'")
            df[col].fillna(val, inplace=True)

    return df


def cap_outliers(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
    inplace: bool = False,
    method: str = "iqr",
) -> pd.DataFrame:
    """
    Cap outliers in numeric columns using IQR/winsorization or percentile clipping.

    Args:
        df: Input dataframe.
        cols: Columns to cap (default: all numeric columns).
        lower_percentile: Used for percentile method; lower cutoff.
        upper_percentile: Used for percentile method; upper cutoff.
        inplace: If True, mutate df in place.
        method: "iqr" (default) or "percentile" for capping.

    Returns:
        DataFrame with outliers capped.
    """
    if not inplace:
        df = df.copy()

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not cols:
        logger.info("No numeric columns to cap outliers.")
        return df

    for col in cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
        elif method == "percentile":
            lower = df[col].quantile(lower_percentile)
            upper = df[col].quantile(upper_percentile)
        else:
            raise ValueError("method must be 'iqr' or 'percentile'")

        before = df[col].copy()
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

        n_capped = (before != df[col]).sum()
        if n_capped > 0:
            logger.info(f"Capped {n_capped} outliers in '{col}' to [{lower:.2f}, {upper:.2f}]")

    return df


def normalize_categories(
    df: pd.DataFrame,
    categories_map: Optional[Dict[str, Dict[Any, Any]]] = None,
    lower: bool = True,
    strip: bool = True,
    one_hot: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Normalize categorical text: lower-case, strip, unify values.
    Optionally apply one-hot encoding.

    Args:
        df: Input dataframe.
        categories_map: Dict of {col: {raw_value: normalized_value}}, to force mapping.
        lower: Lowercase string categories.
        strip: Strip whitespace.
        one_hot: If True, perform pandas one-hot encoding on all categoricals.
        inplace: If True, mutate df in place.

    Returns:
        DataFrame with normalized categories, optionally one-hot encoded.
    """
    if not inplace:
        df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "category", "string"]).columns

    for col in obj_cols:
        df[col] = df[col].astype(str)
        if lower:
            df[col] = df[col].str.lower()
        if strip:
            df[col] = df[col].str.strip()
        if categories_map and col in categories_map:
            logger.info(f"Normalizing '{col}' by explicit mapping: {categories_map[col]}")
            df[col] = df[col].replace(categories_map[col])

    if one_hot:
        logger.info(f"Applying one-hot encoding to categorical columns: {list(obj_cols)}")
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    return df


def dedupe(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Drop duplicate rows from the DataFrame.

    Args:
        df: Input dataframe.
        subset: Columns to consider for finding duplicates (default: all columns).
        keep: Which duplicate to keep: {'first', 'last', False}
        inplace: If True, mutate df in place.

    Returns:
        DataFrame with duplicates removed.
    """
    before = df.shape[0]
    res = df.drop_duplicates(subset=subset, keep=keep, inplace=inplace)
    after = (df if inplace else res).shape[0]
    n_removed = before - after
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate rows.")
    return df if inplace else res


# =====================
# Example Usage
# =====================
if __name__ == "__main__":
    # Test pipeline on dummy data
    data = {
        "A": [1, 2, 2, 2, 100, np.nan, 5],
        "B": [" Cat ", "Dog", "dog", "cat", "dog ", "Dog", None],
        "C": [np.nan, 3, 3, 7, 8, 8, 9],
    }
    df = pd.DataFrame(data)
    logger.info("Original dataframe:\n" + str(df))

    df1 = handle_missing(df)
    logger.info("After handle_missing:\n" + str(df1))

    df2 = cap_outliers(df1)
    logger.info("After cap_outliers:\n" + str(df2))

    category_map = {"B": {"cat": "cat", "dog": "dog"}}
    df3 = normalize_categories(df2, categories_map=category_map)
    logger.info("After normalize_categories:\n" + str(df3))

    df4 = dedupe(df3)
    logger.info("After dedupe:\n" + str(df4))

    print("\n✓ Cleaning pipeline executed successfully.")
