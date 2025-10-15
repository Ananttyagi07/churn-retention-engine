"""
Data Validation Schemas & Drift Checks

- Schema validation using Pydantic for DataFrame row structure and types
- Population Stability Index (PSI) and Kolmogorov–Smirnov (KS) for drift detection
- Supports both feature (column) schema checks and feature drift sanity between reference and new distributions

References: [web:247][web:249][web:145][web:256][web:250]
Drift methods: [web:251][web:254][web:252][web:264][web:257][web:260][web:262][web:263]
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError, Field

from scipy.stats import ks_2samp, chi2_contingency

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# --------------------
# Example: Pydantic Schema for row validation
# --------------------

class InputRowSchema(BaseModel):
    customerid: str
    gender: str
    seniorcitizen: int = Field(ge=0, le=1)
    partner: str
    dependents: str
    tenure: int = Field(ge=0)
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingTV: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float = Field(ge=0)
    totalcharges: float = Field(ge=0)
    churn: Optional[str]  # Allow None for inference

def validate_dataframe_schema(
    df: pd.DataFrame, schema: Type[BaseModel], raise_on_error: bool = False
) -> List[Dict[str, Any]]:
    """
    Validate every row in a DataFrame against the provided Pydantic schema.
    Returns errors per row or empty list.

    Args:
        df: DataFrame to validate
        schema: Pydantic BaseModel class
        raise_on_error: If True, throws on first error

    Returns:
        List of dicts: [{'row': i, 'errors': [...]}, ...]
    """
    errors = []
    for i, record in enumerate(df.to_dict(orient="records")):
        try:
            schema(**record)
        except ValidationError as ve:
            err_detail = {"row": i, "errors": ve.errors()}
            errors.append(err_detail)
            if raise_on_error:
                raise ve
    return errors

# --------------------
# Population Stability Index (PSI)
# --------------------
def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI) between two numerical distributions.

    Args:
        expected: Reference population (training) data
        actual: New/production/validation data
        bins: Number of bins

    Returns:
        PSI value (0-1+), higher=more drift[web:251][web:260][web:262][web:264]
    """
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.percentile(expected, quantiles * 100))
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts = np.histogram(actual, breakpoints)[0]

    expected_ratio = expected_counts / (expected_counts.sum() + 1e-8)
    actual_ratio = actual_counts / (actual_counts.sum() + 1e-8)

    psi = np.sum(
        (expected_ratio - actual_ratio)
        * np.log((expected_ratio + 1e-8) / (actual_ratio + 1e-8))
    )
    return psi

# --------------------
# Feature Drift Detection
# --------------------

def feature_drift_report(
    ref_df: pd.DataFrame,
    new_df: pd.DataFrame,
    feature_types: Optional[Dict[str, str]] = None,
    psi_threshold: float = 0.2,
    p_value_threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    For each column, run numeric (PSI, KS) or categorical (Chi-square) drift tests.
    Returns a report dict with drift/stability status for each feature.

    Args:
        ref_df: Reference (train) dataframe
        new_df: New data (validation, test, or prod)
        feature_types: Dict {column: "numeric"/"categorical"}, otherwise inferred
        psi_threshold: Above this, feature considered drifted
        p_value_threshold: If p < this (KS/chi2), feature considered drifted

    Returns:
        Dict: {
            colname: {
                'type': ...,
                'psi': ...,
                'ks_pvalue': ...,
                'chi2_pvalue': ...,
                'drifted': True/False,
            }, ...
        }
    """
    report = {}
    all_columns = set(ref_df.columns) & set(new_df.columns)
    for col in all_columns:
        col_type = None
        psi, ks_p, chi2_p = None, None, None

        # Detect type if not provided
        if feature_types and col in feature_types:
            col_type = feature_types[col]
        else:
            if np.issubdtype(ref_df[col].dtype, np.number):
                col_type = "numeric"
            else:
                col_type = "categorical"
        res = {"type": col_type, "drifted": False}

        if col_type == "numeric":
            # PSI
            psi = population_stability_index(ref_df[col].dropna().values, new_df[col].dropna().values)
            # KS
            ks_stat, ks_p = ks_2samp(ref_df[col].dropna(), new_df[col].dropna())

            res.update({"psi": psi, "ks_pvalue": ks_p})

            if psi > psi_threshold or (ks_p is not None and ks_p < p_value_threshold):
                res["drifted"] = True

        elif col_type == "categorical":
            c1 = ref_df[col].astype(str).value_counts(normalize=True)
            c2 = new_df[col].astype(str).value_counts(normalize=True)
            # Chi-square test
            cats = list(set(c1.index).union(set(c2.index)))
            obs = np.array([c1.reindex(cats, fill_value=0), c2.reindex(cats, fill_value=0)])
            try:
                chi2_stat, chi2_p, _, _ = chi2_contingency(obs)
            except Exception:
                chi2_p = 1.0  # fail safe

            res.update({"chi2_pvalue": chi2_p})
            if chi2_p < p_value_threshold:
                res["drifted"] = True

        report[col] = res
    return report

# ============= Example Usage ===============
if __name__ == "__main__":
    # Schema validation test
    import pandas as pd

    df = pd.DataFrame([
        dict(customerid="abc", gender="male", seniorcitizen=0, partner="no", dependents="no", tenure=12,
             phoneservice="yes", multiplelines="no", internetservice="fiber optic", onlinesecurity="no",
             onlinebackup="yes", deviceprotection="no", techsupport="no", streamingTV="no",
             streamingmovies="no", contract="month-to-month", paperlessbilling="yes", paymentmethod="credit card",
             monthlycharges=20.0, totalcharges=240.0, churn="no")
    ])
    errs = validate_dataframe_schema(df, InputRowSchema)
    print("Schema Errors:", errs)

    # Drift detection example
    ref = pd.Series(np.random.normal(0, 1, 1000))
    new = pd.Series(np.random.normal(0.3, 1.2, 800))
    psi = population_stability_index(ref, new)
    print(f"PSI: {psi:.3f}")

    # Feature drift report
    dfr = pd.DataFrame({
        "x": np.random.normal(0, 1, 500),
        "y": np.random.choice(["a", "b", "c"], 500, p=[0.5, 0.3, 0.2]),
    })
    dfn = pd.DataFrame({
        "x": np.random.normal(0.2, 1.1, 600),
        "y": np.random.choice(["a", "b", "d"], 600, p=[0.4, 0.3, 0.3]),
    })
    fr = feature_drift_report(dfr, dfn)
    print(fr)