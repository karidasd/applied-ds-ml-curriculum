"""Production Unit & Integration Test Suite.

Tests data loading, temporal splitting, anti-leakage guards, transformer invariants,
model calibration, cost-benefit threshold optimization, and FastAPI REST endpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.data_loader import generate_synthetic_subscription_data, temporal_train_test_split
from src.features import (
    DomainBehavioralTransformer,
    audit_feature_leakage,
    create_preprocessor,
)
from src.model import (
    CostBenefitThresholdOptimizer,
    evaluate_churn_model,
    train_churn_model,
    train_ltv_model,
)


@pytest.fixture(scope="session")
def sample_dataset() -> pd.DataFrame:
    """Provides a deterministic small dataset for fast testing."""
    return generate_synthetic_subscription_data(num_samples=300, random_state=42)


def test_synthetic_data_generation_and_schema(sample_dataset: pd.DataFrame) -> None:
    """Verifies generated data dimensions, columns, and target properties."""
    assert len(sample_dataset) == 300
    expected_cols = [
        "customer_id",
        "signup_date",
        "contract_type",
        "payment_method",
        "internet_service",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_support_tickets",
        "avg_weekly_logins",
        "usage_drop_pct_last_30d",
        "has_premium_support",
        "multi_seat_license",
        "churn",
        "customer_ltv",
    ]
    for col in expected_cols:
        assert col in sample_dataset.columns, f"Missing column: {col}"

    churn_rate = sample_dataset["churn"].mean()
    assert 0.10 <= churn_rate <= 0.40, f"Unrealistic churn rate: {churn_rate:.2f}"
    assert (sample_dataset["customer_ltv"] >= 0).all()


def test_temporal_split_no_leakage(sample_dataset: pd.DataFrame) -> None:
    """Validates that temporal splitting avoids lookahead bias."""
    train_df, test_df = temporal_train_test_split(sample_dataset, cutoff_quantile=0.75)

    assert len(train_df) == int(len(sample_dataset) * 0.75)
    assert len(test_df) == len(sample_dataset) - len(train_df)
    assert train_df["signup_date"].max() <= test_df["signup_date"].min()


def test_domain_behavioral_transformer(sample_dataset: pd.DataFrame) -> None:
    """Validates domain interaction feature generation."""
    transformer = DomainBehavioralTransformer()
    transformed = transformer.fit_transform(sample_dataset)

    new_cols = [
        "charge_to_tenure_ratio",
        "tickets_per_tenure",
        "engagement_index",
        "expected_spend_diff",
    ]
    for col in new_cols:
        assert col in transformed.columns
        assert not transformed[col].isna().any()


def test_preprocessor_pipeline_transformation(sample_dataset: pd.DataFrame) -> None:
    """Checks Scikit-Learn ColumnTransformer pipeline produces clean numeric array."""
    pipeline = create_preprocessor()
    X = sample_dataset.drop(columns=["customer_id", "signup_date", "churn", "customer_ltv"])

    X_transformed = pipeline.fit_transform(X)
    assert isinstance(X_transformed, np.ndarray)
    assert not np.isnan(X_transformed).any()
    assert X_transformed.shape[0] == len(sample_dataset)
    assert X_transformed.shape[1] > 10


def test_audit_feature_leakage_detection(sample_dataset: pd.DataFrame) -> None:
    """Validates that leakage auditor catches synthetic leakage flags."""
    leaked_df = sample_dataset.copy()
    # Artificially inject a high-leakage feature
    leaked_df["leaked_feature"] = leaked_df["churn"] * 0.99

    report = audit_feature_leakage(leaked_df, leaked_df, target_column="churn")
    assert any("leaked_feature" in item for item in report["suspicious_high_correlation"])
    assert any("customer_id" in item for item in report["id_or_timestamp_in_features"])


def test_model_training_and_cost_optimization(sample_dataset: pd.DataFrame) -> None:
    """Validates churn model training, probability calibration, and threshold tuning."""
    train_df, test_df = temporal_train_test_split(sample_dataset, cutoff_quantile=0.70)
    features = [
        c for c in train_df.columns if c not in ["customer_id", "signup_date", "churn", "customer_ltv"]
    ]

    X_train = train_df[features]
    y_train = train_df["churn"].values
    X_test = test_df[features]
    y_test = test_df["churn"].values

    model = train_churn_model(X_train, y_train, calibrate=True)
    probs = model.predict_proba(X_test)[:, 1]

    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    metrics = evaluate_churn_model(model, X_test, y_test, decision_threshold=0.40)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0

    optimizer = CostBenefitThresholdOptimizer(
        intervention_cost=30.0,
        retention_success_prob=0.50,
        average_customer_value=500.0,
    )
    best_t, profit = optimizer.find_optimal_threshold(y_test, probs)
    assert 0.05 <= best_t <= 0.95


def test_ltv_model_training(sample_dataset: pd.DataFrame) -> None:
    """Tests continuous LTV regression training and prediction."""
    features = [
        c
        for c in sample_dataset.columns
        if c not in ["customer_id", "signup_date", "churn", "customer_ltv"]
    ]
    X = sample_dataset[features]
    y_ltv = sample_dataset["customer_ltv"].values

    ltv_pipeline = train_ltv_model(X, y_ltv)
    preds = ltv_pipeline.predict(X)

    assert len(preds) == len(X)
    assert (preds >= 0).all()


def test_fastapi_endpoints() -> None:
    """Tests FastAPI /health, /predict, and /predict/batch using TestClient."""
    with TestClient(app) as client:
        # 1. Health check
        res_health = client.get("/health")
        assert res_health.status_code == 200
        data_health = res_health.json()
        assert data_health["status"] == "healthy"
        assert "churn" in data_health["models_loaded"]

        # 2. Single customer prediction
        cust_payload = {
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "tenure_months": 4,
            "monthly_charges": 92.50,
            "total_charges": 370.00,
            "num_support_tickets": 3,
            "avg_weekly_logins": 4.1,
            "usage_drop_pct_last_30d": 0.35,
            "has_premium_support": 0,
            "multi_seat_license": 0,
        }
        res_pred = client.post("/predict", json=cust_payload)
        assert res_pred.status_code == 200
        data_pred = res_pred.json()
        assert 0.0 <= data_pred["churn_probability"] <= 1.0
        assert data_pred["churn_prediction"] in [0, 1]
        assert data_pred["risk_tier"] in ["Low", "Medium", "High", "Critical"]
        assert data_pred["estimated_lifetime_value"] >= 0.0

        # 3. Batch prediction
        batch_payload = {"customers": [cust_payload, cust_payload]}
        res_batch = client.post("/predict/batch", json=batch_payload)
        assert res_batch.status_code == 200
        data_batch = res_batch.json()
        assert data_batch["total_customers_scored"] == 2
        assert len(data_batch["predictions"]) == 2

        # 4. Input validation failure
        invalid_payload = cust_payload.copy()
        invalid_payload["tenure_months"] = -5  # Out of bounds
        res_invalid = client.post("/predict", json=invalid_payload)
        assert res_invalid.status_code == 422
