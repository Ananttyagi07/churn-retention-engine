"""
Data Splitting Utilities for Train/Val/Test Sets

Provides robust splitting routines for both standard and time-aware scenarios:
- stratified_split: Maintains the target class proportions for imbalanced classification[web:230][web:233][web:234][web:237][web:239][web:240]
- time_aware_split: Ensures all validation/test data is chronologically after the training data, essential for production-like evaluation[web:235][web:236][web:238][web:241][web:244]

Supports: scikit-learn DataFrames/arrays and Pandas tables. Suitable for churn problems or general ML[web:243][web:246].
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: Optional[float] = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Stratified train/validation/test split. Supports 2-way or 3-way splitting.
    Keeps the target class proportions consistent in each split—best for imbalanced classification[web:230][web:233][web:240].

    Args:
        X: Features dataframe or array.
        y: Target vector or series.
        test_size: Fraction for final holdout (test) set (default: 0.2).
        val_size: Fraction for validation set (of *total*, not remaining after test).
                  If None, only returns train/test.
        random_state: RNG seed for deterministic splits.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        (If val_size is None, X_val/y_val will be None)

    Example:
        X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_split(X, y, 0.2, 0.1)
    """
    if val_size is None or val_size == 0:
        # Only 2-way split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_tr, None, X_te, y_tr, None, y_te

    # 3-way split: train/val/test
    test_ratio = test_size
    val_ratio = val_size / (1 - test_ratio)  # fraction of what remains after test

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def time_aware_split(
    df: pd.DataFrame,
    time_column: str = "date",
    split_strategy: str = "train_val_test",
    test_size: float = 0.2,
    val_size: Optional[float] = 0.1,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Split a DataFrame into train (and optional val)/test sets such that test/val is always after train chronologically.
    For non-leaky, future-proof churn or time series prediction[web:235][web:236][web:238].

    Args:
        df: DataFrame with a time column (timestamp or sortable).
        time_column: The column to use for sorting/ordering.
        split_strategy: 'train_val_test' or 'train_test' splits.
        test_size: Fraction for test set, from the end chronologically.
        val_size: If provided, makes a validation set (taken right before test).

    Returns:
        df_train, df_val (opt.), df_test

    Example:
        train, val, test = time_aware_split(df, "start_date", val_size=0.1)
    """
    assert time_column in df.columns, f"Missing time column {time_column}"
    df_sorted = df.sort_values(time_column).reset_index(drop=True)

    n = len(df_sorted)
    test_start = int(np.ceil(n * (1 - test_size)))
    test_end = n

    if val_size is not None and val_size > 0:
        val_start = int(np.ceil(n * (1 - (test_size + val_size))))
        val = df_sorted.iloc[val_start:test_start]
        train = df_sorted.iloc[:val_start]
        test = df_sorted.iloc[test_start:test_end]
        return train, val, test

    # 2-way (no validation)
    train = df_sorted.iloc[:test_start]
    test = df_sorted.iloc[test_start:test_end]
    return train, None, test

# ===================================
# Example Usage
# ===================================
if __name__ == "__main__":
    # Test stratified
    df = pd.DataFrame({
        'f1': np.random.randn(1000),
        'f2': np.random.randn(1000),
        'churn': np.random.choice([0, 1], size=1000, p=[0.8, 0.2]),
    })
    X = df[['f1', 'f2']]
    y = df['churn']
    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_split(X, y, test_size=0.2, val_size=0.1)
    print(f"Stratified: train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")

    # Test time-aware
    dft = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000),
        'value': np.random.randn(1000),
        'y': np.random.choice([0, 1], size=1000, p=[0.8, 0.2]),
    })
    train, val, test = time_aware_split(dft, "timestamp", val_size=0.1)
    print(f"Time-aware: train {train.shape}, val {val.shape}, test {test.shape}")
