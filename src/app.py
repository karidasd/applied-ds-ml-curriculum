"""Production FastAPI Microservice for Churn and Lifetime Value (LTV) Inference.

Exposes RESTful endpoints for real-time customer scoring, batch inference,
and model health telemetry with Pydantic v2 validation.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.data_loader import generate_synthetic_subscription_data
from src.model import CostBenefitThresholdOptimizer, train_churn_model, train_ltv_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global model state
models: Dict[str, Any] = {}
threshold_config: Dict[str, float] = {"optimal_threshold": 0.35}


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
                "tenure_months": 8,
                "monthly_charges": 89.50,
                "total_charges": 716.00,
                "num_support_tickets": 4,
                "avg_weekly_logins": 5.2,
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


class HealthStatusResponse(BaseModel):
    """API health check response."""

    status: str
    models_loaded: List[str]
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading or bootstrapping models upon startup."""
    logger.info("Initializing ML models for inference service...")
    models_dir = Path("models")
    churn_model_path = models_dir / "churn_pipeline.joblib"
    ltv_model_path = models_dir / "ltv_pipeline.joblib"

    if churn_model_path.exists() and ltv_model_path.exists():
        import joblib

        models["churn"] = joblib.load(churn_model_path)
        models["ltv"] = joblib.load(ltv_model_path)
        logger.info("Loaded persisted models from %s", models_dir.resolve())
    else:
        logger.info("No saved artifacts found. Bootstrapping models on synthetic dataset...")
        df_sample = generate_synthetic_subscription_data(num_samples=1200, random_state=42)
        X = df_sample.drop(columns=["customer_id", "signup_date", "churn", "customer_ltv"])
        y_churn = df_sample["churn"].values
        y_ltv = df_sample["customer_ltv"].values

        models["churn"] = train_churn_model(X, y_churn, calibrate=True)
        models["ltv"] = train_ltv_model(X, y_ltv)

        optimizer = CostBenefitThresholdOptimizer(
            intervention_cost=40.0,
            retention_success_prob=0.55,
            average_customer_value=550.0,
        )
        probs = models["churn"].predict_proba(X)[:, 1]
        best_t, _ = optimizer.find_optimal_threshold(y_churn, probs)
        threshold_config["optimal_threshold"] = best_t
        logger.info("Bootstrapping complete. Optimal threshold set to %.3f", best_t)

    yield
    logger.info("Shutting down ML inference service.")
    models.clear()


app = FastAPI(
    title="Customer Churn & LTV Scoring API",
    description="Production-grade ML inference API serving churn probabilities and LTV forecasts.",
    version="1.0.0",
    lifespan=lifespan,
)


def _determine_risk_tier(prob: float) -> Literal["Low", "Medium", "High", "Critical"]:
    if prob < 0.20:
        return "Low"
    elif prob < 0.45:
        return "Medium"
    elif prob < 0.70:
        return "High"
    return "Critical"


def _score_single(profile: CustomerProfile) -> ChurnPredictionResponse:
    if "churn" not in models or "ltv" not in models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not initialized.",
        )

    df_in = pd.DataFrame([profile.model_dump()])
    churn_prob = float(models["churn"].predict_proba(df_in)[0, 1])
    ltv_pred = float(models["ltv"].predict(df_in)[0])

    t = threshold_config["optimal_threshold"]
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


@app.get("/health", response_model=HealthStatusResponse, tags=["Monitoring"])
def health_check() -> HealthStatusResponse:
    """Service health and model readiness probe."""
    return HealthStatusResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        version="1.0.0",
    )


@app.post("/predict", response_model=ChurnPredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerProfile) -> ChurnPredictionResponse:
    """Calculates churn risk probability, decision thresholding, and LTV for a customer."""
    return _score_single(customer)


@app.post("/predict/batch", response_model=BatchInferenceResponse, tags=["Inference"])
def predict_batch(batch: BatchInferenceRequest) -> BatchInferenceResponse:
    """Batch customer inference endpoint for high-volume scoring."""
    predictions = [_score_single(cust) for cust in batch.customers]
    return BatchInferenceResponse(
        predictions=predictions,
        total_customers_scored=len(predictions),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
