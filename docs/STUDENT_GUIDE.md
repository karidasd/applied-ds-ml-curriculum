# Student & Educator Pedagogical Guide
## Applied Data Science & Machine Learning: Customer Churn & LTV Prediction

> **Welcome Students!**  
> This curriculum bridges the gap between academic theory, Kaggle-style data science, and **real-world production Machine Learning & MLOps Engineering**. In production, models succeed not just by achieving a high evaluation score, but by maintaining robust anti-leakage pipelines, explainable predictions, financial expected-value optimization, containerized serving, and proactive post-deployment drift monitoring.

---

## 🎯 The Core Business Case Study: CloudStream SaaS

You are working as the Principal Machine Learning Engineer at **CloudStream**, a subscription software platform. CloudStream faces customer attrition, but solving this is **not** a simple binary classification problem.

### The Dual-Objective Retention Framework
Retention teams cannot intervene on every single customer who might churn—intervention costs money ($40 per outreach/discount). Therefore, we need two models working in synergy:
1. **Churn Probability Classifier ($P(\text{churn} = 1)$):** Estimates the likelihood that a customer cancels.
2. **Continuous LTV Regressor ($\widehat{\text{LTV}}$):** Forecasts the residual monetary worth of the customer if retained.

$$\text{Intervention Priority Score} = P(\text{churn}) \times \widehat{\text{LTV}}$$

---

## 🧭 Complete 5-Module Curriculum Roadmap

```mermaid
flowchart TD
    M1["Module 1: Exploratory Analytics & DuckDB<br/>(Find Cohort & Usage Behavioral Signals)"] --> M2["Module 2: Feature Engineering & Anti-Leakage<br/>(Temporal Split & Scikit-Learn Custom Transformers)"]
    M2 --> M3["Module 3: Imbalanced ML & Expected Value<br/>(CalibratedClassifierCV & Financial Profit Curve)"]
    M3 --> M4["Module 4: Explainable AI (XAI) & Playbooks<br/>(TreeSHAP, Local Waterfall & Retention Actions)"]
    M4 --> M5["Module 5: Post-Deployment Drift Monitoring<br/>(Kolmogorov-Smirnov & PSI Retraining Triggers)"]
    M5 --> M6["Production MLOps: FastAPI, Docker & CI/CD<br/>(Containerized REST API, Pytest, Benchmarks)"]
```

---

### Module 1: In-Memory Analytical SQL with DuckDB
**File:** `notebooks/01_eda_and_duckdb_analytics.ipynb`

#### Why DuckDB?
Traditional Data Science curricula teach Pandas for exploratory data analysis. While Pandas is versatile, in enterprise environments data lives in data warehouses (Snowflake, BigQuery). **DuckDB** brings column-store SQL performance directly into Python notebooks with zero setup overhead and sub-second execution over millions of rows.

#### Key Student Concepts:
1. **Cohort Survival Analysis:** Notice how contract duration dictates churn:
   - Month-to-Month contracts: **~42.5% churn** (high vulnerability).
   - One-Year contracts: **~11.8% churn**.
   - Two-Year contracts: **~3.2% churn** (strong retention moat).
2. **Behavioral Leading Indicators (Velocity):** Churn does not happen overnight. Customers with a **>35% drop in 30-day activity** exhibit a **64.8% churn rate**. This behavioral decay is our primary predictive signal.

---

### Module 2: Feature Engineering & Anti-Leakage Architecture
**File:** `notebooks/02_feature_engineering_and_leakage.ipynb`

#### What is Data Leakage?
Data leakage occurs when information from the future (or the target itself) contaminates the training feature space, giving an illusion of high performance that crashes in production.

#### The Three Deadly Sins of Leakage:
1. **Lookahead Bias (Temporal Bleed):**
   - *Wrong:* Using `train_test_split(shuffle=True)` on subscription events.
   - *Right:* Strictly chronological partitioning: `train_df['date'].max() <= test_df['date'].min()`.
2. **Preprocessor Contamination:**
   - *Wrong:* Calling `StandardScaler.fit_transform(full_df)` before splitting.
   - *Right:* Fitting the transformer **strictly on the training split** and calling `.transform()` on holdout/serving sets.
3. **Target Proxy Leakage:**
   - Features populated only during or after the cancellation event (e.g. `cancellation_reason`, `exit_survey_score`).

#### Writing Modular Scikit-Learn Transformers:
Students implement the `DomainBehavioralTransformer`, inheriting from `BaseEstimator` and `TransformerMixin`:
```python
class DomainBehavioralTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Stateless transformer
        
    def transform(self, X):
        X_out = X.copy()
        # Compute ticket velocity & engagement index
        X_out['ticket_velocity'] = X_out['num_support_tickets'] / (X_out['tenure_months'] + 1.0)
        return X_out
```

---

### Module 3: Imbalanced ML, Calibration & Expected Value Optimization
**File:** `notebooks/03_imbalanced_ml_and_evaluation.ipynb`

#### 1. Why Accuracy is a Trap in Imbalanced ML
In an 80/20 class imbalance, a dummy model that predicts "Nobody Churns" achieves **80% accuracy** while failing 100% of the business objectives.
- Evaluate with **PR-AUC (Average Precision)** and **ROC-AUC**.
- Monitor the **Brier Score** to measure probabilistic error.

#### 2. The Science of Probability Calibration
Tree-based ensembles (Gradient Boosting, XGBoost, Random Forests) push predictions away from 0 and 1, distorting the true posterior probabilities. We wrap our model in `CalibratedClassifierCV(method='isotonic')`:
> **Rule of Calibration:** If the model assigns a churn probability of $0.70$ to 100 customers, approximately 70 of them must actually cancel.

#### 3. The Business Profit Curve (Threshold Tuning)
Defaulting to an arbitrary threshold of $t = 0.50$ is almost always suboptimal. We define the **Expected Value Matrix**:

| Model Decision | Ground Truth: Churns | Ground Truth: Retained |
|---|---|---|
| **Intervene ($\widehat{y}=1$)** | **True Positive (TP)**<br/>Net Gain: $p_{\text{save}} \times \text{LTV} - C$ | **False Positive (FP)**<br/>Net Loss: $-C$ (wasted outreach) |
| **No Action ($\widehat{y}=0$)** | **False Negative (FN)**<br/>Lost Customer: $\$0$ saved | **True Negative (TN)**<br/>Optimal: $\$0$ spent |

**Financial Net Profit Equation:**
$$\text{Net Profit}(t) = \text{TP}(t) \cdot \Big(p_{\text{save}} \cdot \text{Value}_{\text{avg}} - C_{\text{intervention}}\Big) - \text{FP}(t) \cdot C_{\text{intervention}}$$

By evaluating across candidate thresholds $t \in [0.05, 0.95]$, students discover the profit-maximizing threshold **$t^* = 0.35$**, generating thousands of dollars in commercial uplift over the naive $0.50$ baseline!

---

### Module 4: Explainable AI (XAI) with SHAP & Business Decision Playbooks
**File:** `notebooks/04_xai_shap_and_business_thresholds.ipynb`

#### Why Black-Box Predictions Fail in Operations:
A prediction of `P(churn) = 0.82` tells a Customer Success Manager that the customer is in danger, but offers no indication of what action to take. Applying a blunt 20% discount to an account that is frustrated by unresolved bugs is a waste of margin.

#### TreeSHAP Foundations:
Derived from cooperative game theory, Shapley values quantify the exact additive contribution of each feature to pushing the model output from the base value $E[f(x)]$ to the final prediction $f(x)$.

#### The Actionable Retention Matrix:
| Dominant SHAP Driver | Diagnostic Interpretation | Automated Operational Playbook |
|---|---|---|
| `usage_drop_pct_last_30d` | Behavioral workflow abandonment | In-app product re-engagement guide & adoption review |
| `num_support_tickets` | Technical friction / unresolved bugs | Immediate escalation call with Senior Customer Support |
| `contract_type_Month-to-month` | No contractual switching barrier | 15% discount offer for 1-year commitment upgrade |
| `monthly_charges` | Pricing sensitivity | Personalized enterprise pricing bundle restructure |

---

### Module 5: Post-Deployment Data Drift Detection & Model Monitoring
**File:** `notebooks/05_drift_detection_and_monitoring.ipynb`

#### The Types of Production Shift:
1. **Covariate Shift / Data Drift:** $P(X)$ changes while $P(Y|X)$ remains constant (e.g. demographic influx).
2. **Prior Probability Shift:** $P(Y)$ changes (e.g. seasonal renewal cycles).
3. **Concept Drift:** $P(Y|X)$ changes (e.g. a competitor enters the market; customer churn propensity changes fundamentally).

#### Statistical Test Tooling:
- **Two-Sample Kolmogorov-Smirnov (KS) Test:** Non-parametric test comparing the cumulative distribution functions of continuous variables. If $p < 0.05$, reject $H_0$ (distribution shift detected).
- **Population Stability Index (PSI):**
  $$\text{PSI} = \sum_{i=1}^{k} (P_i - Q_i) \cdot \ln\left(\frac{P_i}{Q_i}\right)$$
  - $\text{PSI} < 0.10$: Stable
  - $0.10 \le \text{PSI} < 0.25$: Moderate drift (investigate)
  - $\text{PSI} \ge 0.25$: Significant drift (triggers automated retraining)

---

## ⚡ Ingestion & Query Benchmarks: Pandas vs DuckDB vs Polars
**File:** `src/benchmark.py`

Students benchmark analytical throughput across data engines over $N = 25,000$ to $250,000$ records:
- **DuckDB Zero-Copy SQL:** $3.5\times$ to $5.2\times$ faster CSV scanning with near-zero memory footprint.
- **Polars / Arrow Vectorization:** Fast multi-threaded aggregation for complex grouping operations.
- Run anytime via:
  ```bash
  python -m src.benchmark --samples 25000 --runs 3
  ```

---

## 🐳 Containerization & Zero-Setup DevContainers
- **GitHub Codespaces / VS Code DevContainers (`.devcontainer/devcontainer.json`):** One-click, browser-based cloud environment with Python 3.11, Docker, Jupyter, and Ruff pre-configured.
- **Production Docker (`Dockerfile` & `docker-compose.yml`):**
  ```bash
  docker compose up --build
  ```
  Launches:
  - `applied-ds-ml-api` at `http://localhost:8000` (FastAPI inference engine)
  - `applied-ds-ml-jupyter` at `http://localhost:8888` (Interactive notebook lab)

---

## 📝 Student Self-Assessment Questions

1. **Why is a temporal train/test split mandatory for subscription telemetry compared to random $k$-fold cross validation?**
2. **What occurs to the False Positive Rate when we lower the classification decision threshold from $0.50$ to $0.35$? Why does this increase net profit in our case study?**
3. **Explain the difference between Platt Scaling (sigmoid) and Isotonic Regression in probability calibration.**
4. **How does pairing continuous LTV regression with binary churn classification prevent marketing budget waste?**
5. **How does SHAP feature attribution differ from traditional Gini impurity feature importance?**
6. **When the Population Stability Index (PSI) exceeds 0.25 on production telemetry, what operational steps should an MLOps engineer take?**

---
*Happy Modeling! Built to empower students, researchers, and recruiters with production-grade ML excellence.*
