# Models

The notebooks create the model artifacts in this directory.

Expected files after running the notebooks:

- `churn_model.joblib` — trained XGBoost model
- `feature_columns.joblib` — transformed feature column order
- `model_info.joblib` — metadata used by the Flask application

The repository does not commit a pre-trained binary model because the dataset is obtained from a third-party public source at runtime. Run `01_churn_prediction.ipynb` before using the app for a reproducible local model.
