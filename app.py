from flask import Flask, render_template, request
import os
import urllib.request
import joblib
import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

app = Flask(__name__)
MODEL_PATH = "models/churn_model.joblib"
DATA_PATH = "data/Telco-Customer-Churn.csv"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

FIELDS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}

DEFAULTS = {
    "gender": "Male", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No", "tenure": "12",
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": "75", "TotalCharges": "900"
}


def train_if_needed():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    X = df.drop(columns=["Churn", "customerID"])
    y = (df["Churn"] == "Yes").astype(int)
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", "passthrough", numeric),
    ])
    model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                          subsample=0.85, colsample_bytree=0.85,
                          objective="binary:logistic", eval_metric="logloss",
                          random_state=42, n_jobs=4)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def build_customer(form):
    row = {}
    for field in FIELDS:
        value = form.get(field, DEFAULTS[field]).strip()
        if field in {"tenure", "MonthlyCharges", "TotalCharges"}:
            row[field] = float(value)
        elif field == "SeniorCitizen":
            row[field] = value
        else:
            row[field] = value
    return pd.DataFrame([row], columns=FIELDS)


def explain_customer(pipeline, customer):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(customer)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed)
    values = explanation.values[0]
    order = np.argsort(np.abs(values))[::-1][:7]
    factors = []
    for i in order:
        direction = "increases" if values[i] > 0 else "reduces"
        factors.append({"feature": feature_names[i].replace("categorical__", "").replace("numeric__", ""),
                        "value": float(values[i]), "direction": direction})
    return factors


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", categories=CATEGORIES, values=DEFAULTS)
    try:
        customer = build_customer(request.form)
        pipeline = train_if_needed()
        probability = float(pipeline.predict_proba(customer)[0, 1])
        prediction = "Churn" if probability >= 0.5 else "No Churn"
        factors = explain_customer(pipeline, customer)
        risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Lower"
        return render_template("result.html", customer=customer.iloc[0].to_dict(), probability=probability,
                               prediction=prediction, risk=risk, factors=factors)
    except Exception as exc:
        return render_template("result.html", error=str(exc), customer={}, probability=0,
                               prediction="Unavailable", risk="Unavailable", factors=[])


if __name__ == "__main__":
    app.run(debug=True)
