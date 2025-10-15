"""
Feature Engineering Module

Functions:
- build_behavioral_features(): Add features representing customer usage trends, ticket frequencies
- build_temporal_features(): Extract tenure deltas, recency, seasonality and time-derived features
- encode_categoricals(): Robust one-hot/target encoding, with rare category handling
- scale_features(): Standardize or robust-scale features and save scaler objects for inference integrity
- balance_classes(): Resample data for class balance using SMOTE/ADASYN

References: [web:210][web:212][web:217][web:219][web:218][web:223][web:224][web:226][web:227][web:228]
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any

from loguru import logger
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN

# ================================
# Behavioral Feature Engineering
# ================================

def build_behavioral_features(
    df: pd.DataFrame,
    usage_cols: Optional[List[str]] = None,
    ticket_col: Optional[str] = "supporttickets",
    window: int = 3
) -> pd.DataFrame:
    """
    Add behavioral features such as moving averages, trends, and support ticket frequencies.

    Args:
        df: Input dataframe
        usage_cols: List of columns corresponding to quantitative service usage (e.g., 'minutes', 'usage')
        ticket_col: Support ticket count column (optional)
        window: Period for trend calculations

    Returns:
        DataFrame with new behavioral features
    """
    df = df.copy()
    logger.info("Adding behavioral features...")
    if usage_cols:
        for col in usage_cols:
            df[f"{col}_avg_last_{window}"] = (
                df.groupby("customerid")[col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f"{col}_change_{window}"] = (
                df.groupby("customerid")[col]
                .transform(lambda x: x.diff(window))
            )
    if ticket_col and ticket_col in df.columns:
        df["ticket_freq_per_month"] = (
            df[ticket_col] / df["tenure"].replace(0, np.nan)
        )
        df["ticket_freq_per_month"] = df["ticket_freq_per_month"].fillna(0)

    return df

# ================================
# Temporal Features
# ================================

def build_temporal_features(
    df: pd.DataFrame,
    tenure_col: str = "tenure",
    last_interaction_col: str = "lastinteractiondate",
    contract_col: str = "contract"
) -> pd.DataFrame:
    """
    Extract temporal features: progression of tenure, recency of activity, seasonality bucket.

    Args:
        df: Input dataframe
        tenure_col: Name of tenure column (months)
        last_interaction_col: Date of last interaction (YYYY-MM-DD string)
        contract_col: Type of contract (optional)

    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    logger.info("Building temporal features...")

    # Tenure buckets/labels
    bins = [0, 6, 12, 24, 36, 60, np.inf]
    labels = ['0-6m', '7-12m', '13-24m', '25-36m', '37-60m', '60m+']
    df["tenure_bucket"] = pd.cut(df[tenure_col], bins, labels=labels, right=True)

    # Recency: months since last interaction
    if last_interaction_col in df.columns:
        df[last_interaction_col] = pd.to_datetime(df[last_interaction_col], errors='coerce')
        df["recency_months"] = (
            (pd.Timestamp.now() - df[last_interaction_col]).dt.days / 30.0
        )

    # Seasonality: join/interaction month, quarter
    if last_interaction_col in df.columns:
        df["last_month"] = df[last_interaction_col].dt.month
        df["last_quarter"] = df[last_interaction_col].dt.quarter

    # Contract duration (if available)
    if contract_col in df.columns:
        # Month-to-month = 1, one year = 12, two year = 24
        df["contract_months"] = (
            df[contract_col]
            .map({"month-to-month": 1, "one year": 12, "two year": 24})
            .fillna(1)
        )

    return df

# ================================
# Categorical Encoding
# ================================

def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    encoding: str = "onehot",
    min_freq: float = 0.01,
    target: Optional[pd.Series] = None,
    return_transformer: bool = False,
) -> Tuple[pd.DataFrame, Optional[Any]]:
    """
    Encode categorical features using one-hot, label, or target encoding.
    Groups rare categories based on min_freq.

    Args:
        df: Input dataframe
        categorical_cols: Columns to encode (if None, all object/category/string)
        encoding: 'onehot', 'label', or 'target'
        min_freq: Minimum frequency for rare handling (e.g., 0.01 to group <1% categories as "rare")
        target: For target encoding (must be provided)
        return_transformer: If True, return fitted transformer for inference

    Returns:
        Tuple: (encoded DataFrame, transformer if requested)
    """
    df = df.copy()
    logger.info(f"Encoding categoricals using {encoding} encoding ...")
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Rare category consolidation
    for col in categorical_cols:
        freq = df[col].value_counts(normalize=True)
        rare_labels = freq[freq < min_freq].index
        df[col] = df[col].where(~df[col].isin(rare_labels), other="rare")
        logger.debug(f"Reassigned {len(rare_labels)} rare categories in '{col}' to 'rare'")

    transformer = None
    if encoding == "onehot":
        transformer = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded = transformer.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=transformer.get_feature_names_out(categorical_cols),
            index=df.index,
        )
        df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    elif encoding == "label":
        df_encoded = df.copy()
        transformer = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
        for col, le in transformer.items():
            df_encoded[col] = le.transform(df[col])
    elif encoding == "target":
        # Requires a target column (regression/classification)
        if target is None:
            raise ValueError("Target encoding requires target variable.")
        df_encoded = df.copy()
        for col in categorical_cols:
            means = target.groupby(df[col]).mean()
            df_encoded[col] = df[col].map(means).fillna(means.mean())
    else:
        raise ValueError("encoding must be one of ['onehot', 'label', 'target']")

    logger.info(f"Categorical encoding complete. Shape: {df_encoded.shape}")
    if return_transformer:
        return df_encoded, transformer
    return df_encoded, None

# ================================
# Feature Scaling & Transformations
# ================================

def scale_features(
    df: pd.DataFrame,
    scaling: str = "standard",
    numerical_cols: Optional[List[str]] = None,
    return_scaler: bool = False,
) -> Tuple[pd.DataFrame, Optional[Any]]:
    """
    Apply scaling transformations (standard/robust) to numeric features.

    Args:
        df: Input dataframe
        scaling: "standard" (z-score), "robust" (median/IQR), "minmax"
        numerical_cols: List of numeric columns (if None, auto-detect)
        return_scaler: Whether to return the fitted scaler for inference

    Returns:
        Tuple: (scaled DataFrame, scaler object if requested)
    """
    df = df.copy()
    logger.info(f"Scaling features using {scaling}...")
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = None

    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    elif scaling == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling option {scaling}")

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    logger.info("Scaling complete.")
    if return_scaler:
        return df, scaler
    return df, None

# ================================
# Class Balancing
# ================================

def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote",
    random_state: int = 42,
    return_sampler: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    """
    Oversample the minority class in imbalanced data using SMOTE or ADASYN.

    Args:
        X: Feature matrix (numpy array or DataFrame)
        y: Target vector
        method: 'smote' or 'adasyn'
        random_state: For reproducibility
        return_sampler: If True, returns the fitted sampler object
        kwargs: Additional args for sampler

    Returns:
        Tuple: (X_res, y_res, sampler if requested)
    """
    logger.info(f"Balancing classes using {method.upper()}...")
    if method.lower() == "smote":
        sampler = SMOTE(random_state=random_state, **kwargs)
    elif method.lower() == "adasyn":
        sampler = ADASYN(random_state=random_state, **kwargs)
    else:
        raise ValueError("method must be one of ['smote', 'adasyn']")

    X_res, y_res = sampler.fit_resample(X, y)
    logger.info(f"Resampled class distribution: {np.bincount(y_res) if hasattr(y_res, 'dtype') and np.issubdtype(y_res.dtype, np.integer) else pd.Series(y_res).value_counts().to_dict()}")
    if return_sampler:
        return X_res, y_res, sampler
    return X_res, y_res, None

# ================================
# Example Usage
# ================================

if __name__ == "__main__":
    # Dummy test dataset
    N = 1000
    rng = np.random.default_rng(0)
    df_demo = pd.DataFrame({
        "customerid": np.random.randint(100, 200, size=N),
        "tenure": rng.integers(1, 60, N),
        "supporttickets": np.random.poisson(0.4, N),
        "internetusage": rng.uniform(50, 200, N),
        "contract": rng.choice(["month-to-month", "one year", "two year"], N, p=[0.7, 0.2, 0.1]),
        "product": rng.choice(["premium", "basic", "plus"], N),
        "churn": rng.binomial(1, 0.2, N),
        "lastinteractiondate": pd.date_range("2023-01-01", periods=N, freq="D"),
    })
    logger.info("Running feature engineering demo...")

    # Behavioral
    df1 = build_behavioral_features(df_demo, usage_cols=["internetusage"])
    logger.info("Behavioral features added.")

    # Temporal
    df2 = build_temporal_features(df1)
    logger.info("Temporal features added.")

    # Categorical
    df3, ohe = encode_categoricals(df2, encoding="onehot")
    logger.info("Categoricals encoded.")

    # Scaling
    df4, scaler = scale_features(df3)
    logger.info("Features scaled.")

    # Balancing
    X = df4.drop(columns=["churn"])
    y = df4["churn"].values
    X_res, y_res, sampler = balance_classes(X.values, y, method="smote")
    logger.info(f"Class-balanced feature set: {X_res.shape}, {y_res.shape}")
    print("✓ Example feature pipeline completed.")
