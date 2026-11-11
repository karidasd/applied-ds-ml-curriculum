"""Production ASGI Application Entrypoint for Customer Retention Microservice.

Configures FastAPI lifespan, CORS, telemetry middleware, and mounts API routers.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import config_registry, model_registry, router
from src.data_loader import generate_synthetic_subscription_data
from src.model import CostBenefitThresholdOptimizer, train_churn_model, train_ltv_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager bootstrapping models and baseline distributions."""
    logger.info("Initializing ML models for inference service...")
    models_dir = Path("models")
    churn_path = models_dir / "churn_pipeline.joblib"
    ltv_path = models_dir / "ltv_pipeline.joblib"

    if churn_path.exists() and ltv_path.exists():
        model_registry["churn"] = joblib.load(churn_path)
        model_registry["ltv"] = joblib.load(ltv_path)
        logger.info("Loaded persisted models from %s", models_dir.resolve())
    else:
        logger.info("No saved artifacts found. Bootstrapping models on synthetic dataset...")
        df_sample = generate_synthetic_subscription_data(num_samples=1500, random_state=42)
        X = df_sample.drop(columns=["customer_id", "signup_date", "churn", "customer_ltv"])
        y_churn = df_sample["churn"].values
        y_ltv = df_sample["customer_ltv"].values

        model_registry["churn"] = train_churn_model(X, y_churn, calibrate=True)
        model_registry["ltv"] = train_ltv_model(X, y_ltv)

        optimizer = CostBenefitThresholdOptimizer(
            intervention_cost=40.0,
            retention_success_prob=0.55,
            average_customer_value=550.0,
        )
        probs = model_registry["churn"].predict_proba(X)[:, 1]
        best_t, _ = optimizer.find_optimal_threshold(y_churn, probs)
        config_registry["optimal_threshold"] = best_t
        config_registry["baseline_features"] = X.copy()
        logger.info("Bootstrapping complete. Optimal threshold set to %.3f", best_t)

    yield
    logger.info("Shutting down ML inference service.")
    model_registry.clear()


app = FastAPI(
    title="Customer Churn & LTV Scoring API",
    description=(
        "Production-grade MLOps inference service serving calibrated churn probabilities, "
        "Customer Lifetime Value (LTV) forecasts, Explainable AI (XAI) feature attributions, "
        "and data drift monitoring."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for modern web/dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measures request processing latency and injects diagnostic headers."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time-Sec"] = f"{process_time:.5f}"
    return response


# Mount API Router
app.include_router(router)


@app.get("/", tags=["Root"])
def root_info() -> JSONResponse:
    """Root service information and documentation link."""
    return JSONResponse(
        content={
            "service": "Customer Churn & LTV Scoring API",
            "version": "2.0.0",
            "status": "online",
            "documentation": "/docs",
            "endpoints": [
                "/predict",
                "/predict/batch",
                "/explain",
                "/telemetry/drift",
                "/health",
                "/ready",
            ],
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
