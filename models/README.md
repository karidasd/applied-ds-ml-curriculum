# Model Artifacts Directory

Serialized Scikit-Learn pipelines (`churn_pipeline.joblib` and `ltv_pipeline.joblib`) are persisted in this directory.

Upon FastAPI startup (`src/app.py`), the service automatically loads persisted artifacts if present, or bootstraps new models dynamically if the folder is empty.
