"""Model Training, Calibration, Cost-Sensitive Optimization, and LTV Prediction.

This module implements:
1. Calibrated Churn Classifier with cost-benefit business profit threshold optimization.
2. Continuous Customer Lifetime Value (LTV) Regressor.
3. Modular scikit-learn Pipelines with model persistence and evaluation telemetry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.features import create_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CostBenefitThresholdOptimizer:
    """Optimizes classification decision threshold based on business financial payoffs.

    Parameters
    ----------
    intervention_cost : float
        Cost of retention campaign per targeted customer (e.g., $50 outreach/discount).
    retention_success_prob : float
        Probability that an intervened at-risk customer accepts and is saved (e.g., 0.60).
    average_customer_value : float
        Gross margin or average annual value saved from preventing churn (e.g., $600).
    """

    def __init__(
        self,
        intervention_cost: float = 50.0,
        retention_success_prob: float = 0.50,
        average_customer_value: float = 600.0,
    ) -> None:
        self.intervention_cost = intervention_cost
        self.retention_success_prob = retention_success_prob
        self.average_customer_value = average_customer_value

    def calculate_net_profit(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
        """Computes total financial net profit for a given classification threshold."""
        y_pred = (y_prob >= threshold).astype(int)

        # TP: Customer actually churns, we intervened
        tp = np.sum((y_true == 1) & (y_pred == 1))
        # FP: Customer would NOT have churned, but we intervened needlessly
        fp = np.sum((y_true == 0) & (y_pred == 1))

        # Financial payoff:
        # Saving a churner generates: (retention_success_prob * avg_value) - intervention_cost
        gain_per_tp = (self.retention_success_prob * self.average_customer_value) - self.intervention_cost
        # False alarm costs only the intervention
        cost_per_fp = -self.intervention_cost

        net_profit = (tp * gain_per_tp) + (fp * cost_per_fp)
        return float(net_profit)

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray, num_points: int = 100
    ) -> Tuple[float, float]:
        """Finds the decision threshold maximizing total net business profit."""
        thresholds = np.linspace(0.05, 0.95, num_points)
        profits = [self.calculate_net_profit(y_true, y_prob, t) for t in thresholds]
        best_idx = int(np.argmax(profits))
        optimal_t = float(thresholds[best_idx])
        max_profit = float(profits[best_idx])

        logger.info(
            "Optimal threshold: %.3f yielding net profit of $%.2f (vs baseline 0.50 profit: $%.2f)",
            optimal_t,
            max_profit,
            self.calculate_net_profit(y_true, y_prob, 0.50),
        )
        return optimal_t, max_profit


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    calibrate: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """Builds and trains an end-to-end churn prediction pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray | pd.Series
        Binary churn target.
    calibrate : bool
        Whether to wrap model in CalibratedClassifierCV.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    Pipeline
        Fitted Scikit-Learn pipeline.
    """
    preprocessor = create_preprocessor()

    base_estimator = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=4,
        random_state=random_state,
    )

    if calibrate:
        classifier = CalibratedClassifierCV(
            estimator=base_estimator,
            method="isotonic",
            cv=3,
        )
    else:
        classifier = base_estimator

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    logger.info("Fitting Churn Classifier Pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Churn Classifier training complete.")
    return pipeline


def evaluate_churn_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    decision_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Evaluates churn model with imbalanced classification metrics and calibration metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= decision_threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "decision_threshold": decision_threshold,
        "classification_report": report,
    }

    logger.info(
        "Evaluation -> ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f | Threshold: %.2f",
        roc_auc,
        pr_auc,
        brier,
        decision_threshold,
    )
    return metrics


def train_ltv_model(
    X_train: pd.DataFrame,
    y_train_ltv: np.ndarray | pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Builds and trains an end-to-end continuous LTV regression pipeline."""
    preprocessor = create_preprocessor()

    regressor = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.07,
        max_depth=4,
        loss="squared_error",
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    logger.info("Fitting Customer Lifetime Value (LTV) Regressor...")
    pipeline.fit(X_train, y_train_ltv)
    logger.info("LTV Regressor training complete.")
    return pipeline


def evaluate_ltv_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test_ltv: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """Evaluates LTV regression model performance."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test_ltv, preds)
    rmse = np.sqrt(mean_squared_error(y_test_ltv, preds))
    r2 = r2_score(y_test_ltv, preds)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
    }

    logger.info("LTV Evaluation -> MAE: $%.2f | RMSE: $%.2f | R2: %.4f", mae, rmse, r2)
    return metrics


def save_model(pipeline: Pipeline, filepath: str) -> None:
    """Saves serialized model to disk."""
    joblib.dump(pipeline, filepath)
    logger.info("Model saved successfully to %s", filepath)


def load_model(filepath: str) -> Pipeline:
    """Loads serialized model from disk."""
    pipeline = joblib.load(filepath)
    logger.info("Model loaded successfully from %s", filepath)
    return pipeline
