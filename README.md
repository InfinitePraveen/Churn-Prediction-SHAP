# Churn Prediction + SHAP

An interview-ready Data Science project that predicts telecom customer churn using **XGBoost** and explains individual predictions using **SHAP (SHapley Additive exPlanations)**.

The project combines machine learning, model evaluation, explainable AI, and a Flask web application into one practical end-to-end project.

## Project Overview

Customer churn prediction is a common machine learning problem in the telecom and subscription industries.

In this project, customer information is used to predict whether a customer is likely to **churn**. The XGBoost classification model provides the prediction, while SHAP is used to understand which customer features contributed to that prediction.

The project demonstrates:

* Exploratory Data Analysis
* Data cleaning and feature preparation
* Categorical and numerical feature handling
* XGBoost classification
* Model evaluation
* SHAP-based model explainability
* Global feature importance
* Individual prediction explanations
* Flask web application deployment

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SHAP
* Jupyter Notebook
* Flask
* HTML
* CSS
* Joblib

## Dataset

This project uses the **IBM Telco Customer Churn** dataset.

The dataset contains customer-level information such as:

* Gender
* Senior citizen status
* Partner status
* Dependents
* Tenure
* Phone service
* Internet service
* Contract type
* Payment method
* Monthly charges
* Total charges
* Churn status

The target variable is:

```text
Churn
```

The dataset is publicly available from IBM's GitHub repository:

https://github.com/IBM/telco-customer-churn-on-icp4d

The notebook can download the dataset automatically when it is not available locally.

## Repository Structure

```text
Churn-Prediction-SHAP/
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   ├── 01_churn_prediction.ipynb
│   └── 02_shap_explainability.ipynb
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE
```

The repository intentionally does **not** contain a `src/` directory, separate preprocessing scripts, utility modules, or configuration modules. The project keeps the implementation simple and notebook-focused.

## Project Workflow

```text
Customer Dataset
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Data Cleaning & Feature Preparation
       │
       ▼
Train/Test Split
       │
       ▼
XGBoost Classification
       │
       ▼
Model Evaluation
       │
       ├───────────────┐
       ▼               ▼
Churn Prediction    SHAP Analysis
                       │
                       ├── Global Importance
                       ├── Beeswarm Plot
                       ├── Feature Contributions
                       └── Individual Explanation
                               │
                               ▼
                        Flask Web Application
```

## Notebooks

### 01_churn_prediction.ipynb

This notebook covers the main machine learning workflow:

1. Import required libraries
2. Download/load the Telco Customer Churn dataset
3. Inspect the dataset
4. Perform exploratory data analysis
5. Handle missing values
6. Prepare categorical and numerical features
7. Encode categorical variables
8. Split the dataset into training and testing sets
9. Train the XGBoost classifier
10. Generate predictions
11. Evaluate the model
12. Save model artifacts for the Flask application

### 02_shap_explainability.ipynb

This notebook focuses on explainable AI.

It demonstrates:

1. Loading the trained model
2. Preparing test data
3. Creating a SHAP TreeExplainer
4. Calculating SHAP values
5. Global feature importance
6. SHAP bar plots
7. SHAP beeswarm plots
8. Individual customer explanations
9. Understanding why a prediction moved toward churn or non-churn

## Why XGBoost?

XGBoost is used because the dataset contains structured/tabular customer information.

Tree-based boosting models can capture nonlinear relationships and feature interactions without requiring feature scaling in the same way that many distance-based or linear models do.

The model is used here as a practical example of binary classification for customer churn.

## Why SHAP?

A machine learning model can tell us:

> "This customer is likely to churn."

But that prediction alone does not explain **why**.

SHAP helps answer questions such as:

* Which features influenced this prediction?
* Which features pushed the prediction toward churn?
* Which features pushed it toward non-churn?
* Which features are generally important across the dataset?

This makes the project more useful for discussing **model interpretability and explainable AI** during interviews.

## Model Evaluation

The project evaluates the classification model using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

These metrics provide different perspectives on classification performance, particularly when the cost of missing potential churn customers may differ from the cost of incorrectly flagging customers.

## Flask Web Application

The project includes a Flask web application that allows users to enter customer information and receive an interactive churn prediction.

The application displays:

* Predicted churn/non-churn class
* Churn probability
* Risk interpretation
* Important SHAP factors for the individual prediction

The application is designed as a simple demonstration interface for showcasing how the trained machine learning model can be used outside a notebook.

## Web Application Preview

The application follows a simple workflow:

```text
Customer Information
        │
        ▼
Submit Prediction Form
        │
        ▼
XGBoost Model
        │
        ▼
Churn Probability
        │
        ▼
SHAP Explanation
        │
        ▼
Prediction Result
```

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/InfinitePraveen/Churn-Prediction-SHAP.git
```

Move into the project directory:

```bash
cd Churn-Prediction-SHAP
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Run:

```text
notebooks/01_churn_prediction.ipynb
```

first.

This notebook performs the main data preparation and model-training workflow.

Then run:

```text
notebooks/02_shap_explainability.ipynb
```

to explore the SHAP explanations.

### 5. Run the Flask application

From the project root:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Interview Discussion Points

This project can be used to discuss several important Data Science concepts during interviews.

### Machine Learning

* Binary classification
* Feature engineering
* Train/test splitting
* Model evaluation
* Class probabilities
* XGBoost

### Explainable AI

* What is model interpretability?
* What are SHAP values?
* Global vs local explanations
* Feature contributions
* TreeExplainer
* Limitations of model explanations

### Deployment

* Flask
* HTML forms
* Model inference
* Loading trained model artifacts
* Connecting a machine learning model to a web interface

## Important SHAP Limitation

SHAP explains how features contributed to the output of the trained model.

A SHAP contribution should **not automatically be interpreted as a causal effect**.

For example, if a feature has a positive SHAP contribution toward churn, this means that the feature contributed to the model's prediction. It does not necessarily mean that changing that feature would directly cause the customer to churn.

## GitHub

GitHub:
https://github.com/InfinitePraveen

## LinkedIn

LinkedIn:
https://www.linkedin.com/in/infinitepraveen/

## Contributing

Contributions, suggestions, bug reports, and improvements are welcome.

Please read:

```text
CONTRIBUTING.md
```

before submitting changes.

## Changelog

Project changes are documented in:

```text
CHANGELOG.md
```

## License

This project is intended for learning, experimentation, and portfolio purposes.

Please review the original dataset repository and its licensing terms before redistributing the raw dataset.

---

**Built as a practical Data Science portfolio project demonstrating churn prediction, XGBoost, SHAP explainability, and Flask deployment.**
