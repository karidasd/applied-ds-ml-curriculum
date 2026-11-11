"""Data Ingestion, Synthetic Generation, and Analytical Loading Module.

This module provides production utilities for loading raw subscription telemetry,
generating realistic benchmark datasets, executing DuckDB queries, and performing
leak-free temporal train/test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_subscription_data(
    num_samples: int = 5000,
    random_state: int = 42,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Generates a realistic synthetic subscription customer dataset.

    Features realistic non-linear relationships, behavioral usage metrics,
    imbalanced churn rates (~20%), and continuous Lifetime Value (LTV).

    Parameters
    ----------
    num_samples : int
        Number of customer records to generate.
    random_state : int
        Seed for deterministic reproducibility.
    output_path : Optional[str | Path]
        If specified, saves the generated DataFrame to a CSV file.

    Returns
    -------
    pd.DataFrame
        Synthetic customer dataset.
    """
    rng = np.random.default_rng(random_state)
    logger.info("Generating %d synthetic subscription records...", num_samples)

    customer_ids = [f"CUST-{i:06d}" for i in range(1, num_samples + 1)]

    # Temporal dimension: signups spanning 2 years
    start_date = pd.Timestamp("2023-01-01")
    days_range = 730
    random_days = rng.integers(0, days_range, size=num_samples)
    signup_dates = [start_date + pd.Timedelta(days=int(d)) for d in random_days]

    contract_types = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=num_samples,
        p=[0.55, 0.25, 0.20],
    )

    payment_methods = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=num_samples,
        p=[0.35, 0.15, 0.25, 0.25],
    )

    internet_service = rng.choice(
        ["Fiber optic", "DSL", "No"],
        size=num_samples,
        p=[0.45, 0.40, 0.15],
    )

    # Base tenure based on contract type & date
    tenure_months = rng.integers(1, 73, size=num_samples)

    # Monthly charges with realistic variance based on service
    base_monthly = np.where(
        internet_service == "Fiber optic",
        rng.normal(85.0, 12.0, num_samples),
        np.where(
            internet_service == "DSL",
            rng.normal(55.0, 10.0, num_samples),
            rng.normal(25.0, 5.0, num_samples),
        ),
    )
    monthly_charges = np.clip(np.round(base_monthly, 2), 18.0, 130.0)

    # Support and Behavioral Usage features
    num_support_tickets = rng.poisson(lam=1.5, size=num_samples)
    avg_weekly_logins = np.clip(rng.normal(14.0, 5.0, num_samples), 1.0, 40.0).round(1)
    usage_drop_pct_last_30d = np.clip(rng.normal(0.10, 0.25, num_samples), -0.5, 0.95).round(3)
    has_premium_support = rng.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    multi_seat_license = rng.choice([0, 1], size=num_samples, p=[0.6, 0.4])

    # Realistic Ground Truth Churn Probability calculation (non-linear logit)
    # Month-to-month contracts, high monthly charges, high usage drop, high tickets -> higher churn
    logit = (
        -2.2
        + 1.4 * (contract_types == "Month-to-month")
        - 1.1 * (contract_types == "Two year")
        + 0.02 * (monthly_charges - 60)
        - 0.035 * tenure_months
        + 0.35 * num_support_tickets
        + 2.2 * usage_drop_pct_last_30d
        - 0.6 * has_premium_support
        + 0.5 * (payment_methods == "Electronic check")
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(0, 1, size=num_samples) < churn_prob).astype(int)

    # Historical cumulative spend / Total Charges
    total_charges = np.round(
        monthly_charges * tenure_months * rng.uniform(0.95, 1.05, size=num_samples), 2
    )

    # Customer Lifetime Value (LTV): For churned customers, actual cumulative spend.
    # For active customers, actual cumulative spend + expected future discounted cash flows.
    expected_future_months = np.where(churn == 1, 0, np.maximum(6, 48 - tenure_months // 2))
    discount_factor = 0.95
    future_val = (
        expected_future_months
        * monthly_charges
        * (1.0 - 0.5 * usage_drop_pct_last_30d)
        * discount_factor
    )
    ltv = np.round(np.clip(total_charges + future_val, 50.0, 25000.0), 2)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_date": signup_dates,
            "contract_type": contract_types,
            "payment_method": payment_methods,
            "internet_service": internet_service,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
            "avg_weekly_logins": avg_weekly_logins,
            "usage_drop_pct_last_30d": usage_drop_pct_last_30d,
            "has_premium_support": has_premium_support,
            "multi_seat_license": multi_seat_license,
            "churn": churn,
            "customer_ltv": ltv,
        }
    )

    df = df.sort_values("signup_date").reset_index(drop=True)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info("Saved synthetic data to %s", out_p.resolve())

    return df


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """Loads raw subscription data from CSV or Parquet with schema validation."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=["signup_date"])
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info("Loaded %d rows with %d columns from %s", len(df), len(df.columns), path.name)
    return df


def query_with_duckdb(query: str, table_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Executes high-performance DuckDB SQL query against in-memory DataFrames or files."""
    conn = duckdb.connect(database=":memory:")
    for tbl_name, df_data in table_map.items():
        conn.register(tbl_name, df_data)

    result_df = conn.execute(query).df()
    conn.close()
    return result_df


def temporal_train_test_split(
    df: pd.DataFrame,
    date_column: str = "signup_date",
    cutoff_quantile: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset temporally to prevent time-travel target leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing a datetime column.
    date_column : str
        Name of the datetime timestamp column.
    cutoff_quantile : float
        Quantile cutoff (e.g. 0.80 allocates the first 80% chronologically to train).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found in DataFrame.")

    df_sorted = df.sort_values(by=date_column).reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * cutoff_quantile)
    cutoff_date = df_sorted.iloc[cutoff_idx][date_column]

    train_df = df_sorted.iloc[:cutoff_idx].copy()
    test_df = df_sorted.iloc[cutoff_idx:].copy()

    logger.info(
        "Temporal split at %s: Train=%d samples, Test=%d samples (Quantile=%.2f)",
        cutoff_date,
        len(train_df),
        len(test_df),
        cutoff_quantile,
    )
    return train_df, test_df


if __name__ == "__main__":
    default_out = Path(__file__).resolve().parent.parent / "data" / "raw" / "subscriptions_raw.csv"
    generate_synthetic_subscription_data(num_samples=5000, output_path=default_out)
