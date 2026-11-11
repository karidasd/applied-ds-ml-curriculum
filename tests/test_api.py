"""Dedicated API and Service Integration Tests.

Tests FastAPI endpoints, Pydantic v2 validation contracts, XAI explanations,
and latency diagnostic headers.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.app import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session-scoped test client managing FastAPI lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer_payload() -> dict:
    """Valid customer request payload."""
    return {
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "tenure_months": 8,
        "monthly_charges": 89.50,
        "total_charges": 716.00,
        "num_support_tickets": 3,
        "avg_weekly_logins": 6.2,
        "usage_drop_pct_last_30d": 0.35,
        "has_premium_support": 0,
        "multi_seat_license": 0,
    }


def test_health_and_readiness_endpoints(client: TestClient) -> None:
    """Verifies service health and k8s readiness endpoints."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    assert "churn" in data_health["models_loaded"]

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json() == {"status": "ready"}


def test_root_endpoint(client: TestClient) -> None:
    """Verifies root route returns documentation links."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "/predict" in data["endpoints"]


def test_single_prediction_endpoint(client: TestClient, valid_customer_payload: dict) -> None:
    """Tests single customer scoring, schema conformity, and latency header."""
    res = client.post("/predict", json=valid_customer_payload)
    assert res.status_code == 200
    assert "X-Process-Time-Sec" in res.headers

    data = res.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_tier"] in ["Low", "Medium", "High", "Critical"]
    assert isinstance(data["retention_action_recommended"], bool)
    assert data["estimated_lifetime_value"] >= 0.0


def test_batch_prediction_endpoint(client: TestClient, valid_customer_payload: dict) -> None:
    """Tests batch scoring with multiple customer records."""
    batch_payload = {"customers": [valid_customer_payload, valid_customer_payload, valid_customer_payload]}
    res = client.post("/predict/batch", json=batch_payload)
    assert res.status_code == 200

    data = res.json()
    assert data["total_customers_scored"] == 3
    assert len(data["predictions"]) == 3


def test_explain_endpoint_xai(client: TestClient, valid_customer_payload: dict) -> None:
    """Tests Explainable AI endpoint returning SHAP attribution and playbook."""
    res = client.post("/explain", json=valid_customer_payload)
    assert res.status_code == 200

    data = res.json()
    assert "churn_probability" in data
    assert "top_risk_drivers" in data
    assert len(data["top_risk_drivers"]) > 0
    assert "recommended_business_playbook" in data

    first_driver = data["top_risk_drivers"][0]
    assert "feature_name" in first_driver
    assert first_driver["impact_direction"] in [
        "Increases Churn Risk",
        "Decreases Churn Risk",
        "Neutral",
    ]


def test_drift_telemetry_endpoint(client: TestClient, valid_customer_payload: dict) -> None:
    """Tests data drift check on customer telemetry batch."""
    customers = [valid_customer_payload.copy() for _ in range(15)]
    batch_payload = {"customers": customers}

    res = client.post("/telemetry/drift", json=batch_payload)
    assert res.status_code == 200
    data = res.json()
    assert "overall_drift_status" in data
    assert data["overall_drift_status"] in ["STABLE", "DRIFT_DETECTED", "WARNING"]
    assert isinstance(data["metrics"], list)


def test_validation_errors(client: TestClient, valid_customer_payload: dict) -> None:
    """Verifies that invalid payloads trigger 422 Unprocessable Entity."""
    # Negative tenure
    bad_payload_1 = valid_customer_payload.copy()
    bad_payload_1["tenure_months"] = -10
    res1 = client.post("/predict", json=bad_payload_1)
    assert res1.status_code == 422

    # Invalid contract type
    bad_payload_2 = valid_customer_payload.copy()
    bad_payload_2["contract_type"] = "Lifetime Permanent"
    res2 = client.post("/predict", json=bad_payload_2)
    assert res2.status_code == 422

    # Monthly charges exceeding max
    bad_payload_3 = valid_customer_payload.copy()
    bad_payload_3["monthly_charges"] = 9999.00
    res3 = client.post("/predict", json=bad_payload_3)
    assert res3.status_code == 422
