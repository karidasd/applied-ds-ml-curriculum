"""Feature Engineering, Preprocessing, and Anti-Leakage Pipeline.

This module encapsulates domain-driven feature transformers for subscription telemetry,
scikit-learn ColumnTransformer pipelines, and strict guards against target leakage.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NUMERICAL_FEATURES: List[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "avg_weekly_logins",
    "usage_drop_pct_last_30d",
]

CATEGORICAL_FEATURES: List[str] = [
    "contract_type",
    "payment_method",
    "internet_service",
]

BINARY_FEATURES: List[str] = [
    "has_premium_support",
    "multi_seat_license",
]


class DomainBehavioralTransformer(BaseEstimator, TransformerMixin):
    """Generates subscription domain ratios and behavioral interaction features.

    Features created:
    - charge_to_tenure_ratio: monthly_charges / (tenure_months + 1)
    - tickets_per_tenure: num_support_tickets / (tenure_months + 1)
    - engagement_index: avg_weekly_logins * (1 - usage_drop_pct_last_30d)
    - expected_cumulative_variance: total_charges - (monthly_charges * tenure_months)
    """

    def __init__(self, add_interaction_terms: bool = True) -> None:
        self.add_interaction_terms = add_interaction_terms

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> DomainBehavioralTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        tenure = X_out["tenure_months"].values
        monthly = X_out["monthly_charges"].values
        total = X_out["total_charges"].values
        tickets = X_out["num_support_tickets"].values
        logins = X_out["avg_weekly_logins"].values
        drop_pct = X_out["usage_drop_pct_last_30d"].values

        # Interaction signals
        X_out["charge_to_tenure_ratio"] = monthly / (tenure + 1.0)
        X_out["tickets_per_tenure"] = tickets / (tenure + 1.0)
        X_out["engagement_index"] = logins * np.maximum(0.0, (1.0 - drop_pct))
        X_out["expected_spend_diff"] = total - (monthly * tenure)

        return X_out


def create_preprocessor() -> Pipeline:
    """Builds a leak-free scikit-learn preprocessing Pipeline.

    Applies DomainBehavioralTransformer followed by ColumnTransformer with
    StandardScaler for continuous features and OneHotEncoder for categoricals.

    Returns
    -------
    Pipeline
        Configured scikit-learn preprocessing pipeline.
    """
    all_numeric = NUMERICAL_FEATURES + [
        "charge_to_tenure_ratio",
        "tickets_per_tenure",
        "engagement_index",
        "expected_spend_diff",
    ]

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline(
        steps=[
            ("domain_features", DomainBehavioralTransformer()),
            ("preprocessor", col_transformer),
        ]
    )
    return full_pipeline


def audit_feature_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "churn",
    correlation_threshold: float = 0.90,
) -> dict[str, list]:
    """Audits feature distributions and correlations to flag potential target leakage.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set.
    test_df : pd.DataFrame
        Validation or holdout set.
    target_column : str
        Target column name.
    correlation_threshold : float
        Absolute Pearson correlation flag threshold.

    Returns
    -------
    dict[str, list]
        Dictionary reporting suspected leaked features or temporal anomalies.
    """
    report: dict[str, list] = {
        "suspicious_high_correlation": [],
        "zero_variance_train": [],
        "id_or_timestamp_in_features": [],
    }

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
        target_series = train_df[target_column]

        for col in numeric_cols:
            corr = train_df[col].corr(target_series)
            if abs(corr) >= correlation_threshold:
                report["suspicious_high_correlation"].append(
                    f"{col} (|r| = {abs(corr):.3f})"
                )

    for col in train_df.columns:
        if col.lower() in ["customer_id", "id", "signup_date", "timestamp"]:
            report["id_or_timestamp_in_features"].append(col)
        if train_df[col].nunique() <= 1:
            report["zero_variance_train"].append(col)

    logger.info("Leakage audit complete. Flags detected: %s", report)
    return report


def save_preprocessor(pipeline: Pipeline, filepath: str) -> None:
    """Serializes fitted pipeline to disk."""
    joblib.dump(pipeline, filepath)
    logger.info("Saved fitted preprocessor pipeline to %s", filepath)


def load_preprocessor(filepath: str) -> Pipeline:
    """Deserializes preprocessor pipeline from disk."""
    pipeline = joblib.load(filepath)
    logger.info("Loaded preprocessor pipeline from %s", filepath)
    return pipeline
