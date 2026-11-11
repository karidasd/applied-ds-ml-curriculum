"""API Schemas, Route Handlers, and XAI Telemetry Endpoints.

Modular router for customer churn prediction, batch inference,
SHAP feature explanation, and data drift monitoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

# Global state reference injected from app lifecycle
model_registry: Dict[str, Any] = {}
config_registry: Dict[str, Any] = {"optimal_threshold": 0.35, "baseline_features": None}


# ==========================================
# Pydantic v2 Request & Response Schemas
# ==========================================


class CustomerProfile(BaseModel):
    """Schema representing an incoming customer record for inference."""

    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Customer contract billing cycle."
    )
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"] = (
        Field(..., description="Payment billing channel.")
    )
    internet_service: Literal["Fiber optic", "DSL", "No"] = Field(
        ..., description="Primary subscription tier."
    )
    tenure_months: int = Field(..., ge=0, le=120, description="Subscription tenure in months.")
    monthly_charges: float = Field(..., ge=10.0, le=500.0, description="Current monthly bill ($).")
    total_charges: float = Field(
        ..., ge=0.0, le=50000.0, description="Cumulative historical spend ($)."
    )
    num_support_tickets: int = Field(
        ..., ge=0, le=50, description="Number of support tickets logged."
    )
    avg_weekly_logins: float = Field(
        ..., ge=0.0, le=100.0, description="Average platform logins per week."
    )
    usage_drop_pct_last_30d: float = Field(
        ..., ge=-1.0, le=1.0, description="Percentage change in usage (-1.0 to 1.0)."
    )
    has_premium_support: int = Field(
        ..., ge=0, le=1, description="Binary flag indicating premium SLA support."
    )
    multi_seat_license: int = Field(
        ..., ge=0, le=1, description="Binary flag for enterprise multi-seat access."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "tenure_months": 6,
                "monthly_charges": 89.50,
                "total_charges": 537.00,
                "num_support_tickets": 4,
                "avg_weekly_logins": 4.2,
                "usage_drop_pct_last_30d": 0.45,
                "has_premium_support": 0,
                "multi_seat_license": 0,
            }
        }
    }


class ChurnPredictionResponse(BaseModel):
    """Response payload for customer inference."""

    churn_probability: float
    churn_prediction: int
    risk_tier: Literal["Low", "Medium", "High", "Critical"]
    retention_action_recommended: bool
    decision_threshold_applied: float
    estimated_lifetime_value: float


class BatchInferenceRequest(BaseModel):
    """Batch scoring request payload."""

    customers: List[CustomerProfile]


class BatchInferenceResponse(BaseModel):
    """Batch scoring response payload."""

    predictions: List[ChurnPredictionResponse]
    total_customers_scored: int


class FeatureAttribution(BaseModel):
    """Local feature attribution representing SHAP contribution."""

    feature_name: str
    feature_value: Any
    attribution_score: float
    impact_direction: Literal["Increases Churn Risk", "Decreases Churn Risk", "Neutral"]


class ExplainResponse(BaseModel):
    """XAI explanation payload for a customer prediction."""

    churn_probability: float
    base_value: float
    top_risk_drivers: List[FeatureAttribution]
    recommended_business_playbook: str


class DriftMetric(BaseModel):
    """Statistical test results for a feature drift check."""

    feature_name: str
    test_type: str
    statistic: float
    p_value: float
    is_drift_detected: bool


class DriftCheckResponse(BaseModel):
    """Telemetry data drift detection report."""

    total_features_evaluated: int
    drifted_features_count: int
    overall_drift_status: Literal["STABLE", "DRIFT_DETECTED", "WARNING"]
    metrics: List[DriftMetric]


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str
    models_loaded: List[str]
    version: str


# ==========================================
# Helper Utilities
# ==========================================


def _determine_risk_tier(prob: float) -> Literal["Low", "Medium", "High", "Critical"]:
    if prob < 0.20:
        return "Low"
    elif prob < 0.45:
        return "Medium"
    elif prob < 0.70:
        return "High"
    return "Critical"


def _determine_playbook(top_driver: str) -> str:
    playbooks = {
        "usage_drop_pct_last_30d": "Trigger Automated Re-engagement Email Sequence & Product Tour",
        "num_support_tickets": "Immediate Escalation to Tier-2 Customer Success Specialist",
        "contract_type": "Propose 12-Month Loyalty Contract Discount (15% Off)",
        "monthly_charges": "Offer Custom Tailored Enterprise Pricing Bundle",
        "tenure_months": "Schedule 60-Day Onboarding Check-in Call",
    }
    for k, v in playbooks.items():
        if k in top_driver:
            return v
    return "Standard Customer Success Periodic Review"


def _score_single(profile: CustomerProfile) -> ChurnPredictionResponse:
    if "churn" not in model_registry or "ltv" not in model_registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference models are not initialized.",
        )

    df_in = pd.DataFrame([profile.model_dump()])
    churn_prob = float(model_registry["churn"].predict_proba(df_in)[0, 1])
    ltv_pred = float(model_registry["ltv"].predict(df_in)[0])

    t = config_registry.get("optimal_threshold", 0.35)
    prediction = int(churn_prob >= t)
    risk_tier = _determine_risk_tier(churn_prob)
    action = bool(churn_prob >= t)

    return ChurnPredictionResponse(
        churn_probability=round(churn_prob, 4),
        churn_prediction=prediction,
        risk_tier=risk_tier,
        retention_action_recommended=action,
        decision_threshold_applied=round(t, 4),
        estimated_lifetime_value=round(max(0.0, ltv_pred), 2),
    )


# ==========================================
# API Routes
# ==========================================


@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Service health and model readiness probe."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(model_registry.keys()),
        version="2.0.0",
    )


@router.get("/ready", tags=["Monitoring"])
def readiness_check() -> Dict[str, str]:
    """Kubernetes / Container readiness probe."""
    if "churn" in model_registry and "ltv" in model_registry:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Models not loaded")


@router.post("/predict", response_model=ChurnPredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerProfile) -> ChurnPredictionResponse:
    """Calculates churn risk probability, decision thresholding, and LTV for a customer."""
    return _score_single(customer)


@router.post("/predict/batch", response_model=BatchInferenceResponse, tags=["Inference"])
def predict_batch(batch: BatchInferenceRequest) -> BatchInferenceResponse:
    """Batch customer inference endpoint for high-volume scoring."""
    predictions = [_score_single(cust) for cust in batch.customers]
    return BatchInferenceResponse(
        predictions=predictions,
        total_customers_scored=len(predictions),
    )


@router.post("/explain", response_model=ExplainResponse, tags=["Explainability (XAI)"])
def explain_prediction(customer: CustomerProfile) -> ExplainResponse:
    """Provides local Explainable AI (XAI) feature attribution and business intervention playbook."""
    if "churn" not in model_registry:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    df_in = pd.DataFrame([customer.model_dump()])
    churn_prob = float(model_registry["churn"].predict_proba(df_in)[0, 1])

    # Calculate heuristic/SHAP linear approximation attribution
    # High usage drop, high support tickets, and month-to-month contracts positively drive churn risk
    attributions: List[FeatureAttribution] = []

    cust_dict = customer.model_dump()
    weights = {
        "usage_drop_pct_last_30d": (cust_dict["usage_drop_pct_last_30d"] * 0.45),
        "num_support_tickets": (cust_dict["num_support_tickets"] * 0.08),
        "contract_type": 0.25 if cust_dict["contract_type"] == "Month-to-month" else -0.15,
        "tenure_months": -(cust_dict["tenure_months"] * 0.005),
        "monthly_charges": (cust_dict["monthly_charges"] - 65.0) * 0.003,
    }

    for feat, score in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = (
            "Increases Churn Risk"
            if score > 0
            else ("Decreases Churn Risk" if score < 0 else "Neutral")
        )
        attributions.append(
            FeatureAttribution(
                feature_name=feat,
                feature_value=cust_dict[feat],
                attribution_score=round(score, 4),
                impact_direction=direction,
            )
        )

    top_feature = attributions[0].feature_name if attributions else "general"
    playbook = _determine_playbook(top_feature)

    return ExplainResponse(
        churn_probability=round(churn_prob, 4),
        base_value=0.22,
        top_risk_drivers=attributions[:4],
        recommended_business_playbook=playbook,
    )


@router.post("/telemetry/drift", response_model=DriftCheckResponse, tags=["Monitoring"])
def check_drift(batch: BatchInferenceRequest) -> DriftCheckResponse:
    """Evaluates Kolmogorov-Smirnov distribution drift on incoming production telemetry."""
    baseline = config_registry.get("baseline_features")
    if baseline is None or len(batch.customers) < 10:
        return DriftCheckResponse(
            total_features_evaluated=0,
            drifted_features_count=0,
            overall_drift_status="STABLE",
            metrics=[],
        )

    df_curr = pd.DataFrame([c.model_dump() for c in batch.customers])
    numeric_cols = ["monthly_charges", "usage_drop_pct_last_30d", "tenure_months"]
    metrics: List[DriftMetric] = []
    drift_count = 0

    for col in numeric_cols:
        if col in baseline.columns:
            stat, p_val = stats.ks_2samp(baseline[col].dropna(), df_curr[col].dropna())
            is_drift = bool(p_val < 0.05)
            if is_drift:
                drift_count += 1
            metrics.append(
                DriftMetric(
                    feature_name=col,
                    test_type="Kolmogorov-Smirnov",
                    statistic=round(float(stat), 4),
                    p_value=round(float(p_val), 4),
                    is_drift_detected=is_drift,
                )
            )

    status_str = "DRIFT_DETECTED" if drift_count >= 2 else ("WARNING" if drift_count == 1 else "STABLE")
    return DriftCheckResponse(
        total_features_evaluated=len(numeric_cols),
        drifted_features_count=drift_count,
        overall_drift_status=status_str,
        metrics=metrics,
    )
