# Churn Prediction + SHAP

An interview-ready data science project that predicts telecom customer churn with **XGBoost** and explains individual and global predictions with **SHAP**.

The project uses IBM's publicly available **Telco Customer Churn** dataset. The dataset is downloaded at runtime from IBM's public GitHub repository rather than redistributed in this repository. The dataset contains customer-level telecom information and a binary `Churn` target.

## What this project demonstrates

- Exploratory data analysis with pandas, matplotlib and seaborn
- Practical handling of categorical and numeric customer data
- XGBoost binary classification
- Stratified train/test split
- ROC-AUC, precision, recall, F1-score and confusion matrix
- SHAP global feature importance
- SHAP beeswarm and bar plots
- Local explanation for an individual customer
- Flask web application for interactive churn prediction
- Human-readable explanation of why a customer received a high/low churn score

SHAP's `TreeExplainer` is designed for tree-based models such as XGBoost and provides feature-level attributions for predictions.

## Dataset

**IBM Telco Customer Churn**

Source: IBM's public repository:
https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

The notebook downloads the CSV automatically when it is not present locally.

## Repository structure

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
└── CHANGELOG.md
```

There is intentionally **no `src/` directory**, no separate preprocessing module and no utility/config module. The notebooks contain the data preparation and model-development workflow, while `app.py` contains the small amount of logic required to serve the trained model.

## How to run

### 1. Install Python

Python 3.11 or newer is recommended.

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Run `01_churn_prediction.ipynb` first. It downloads the IBM dataset, performs EDA, prepares the data, trains XGBoost and saves the trained model metadata under `models/`.

Then run `02_shap_explainability.ipynb` to create the global and local SHAP explanations.

### 5. Start the Flask app

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

If the model files are not present, the application can train a compact model automatically from the public IBM CSV on first launch. For an interview demonstration, it is better to run the notebooks first so the model and explanation workflow are visible.

## Web application

The Flask app provides:

- Customer input form
- Churn probability
- Predicted churn/non-churn class
- Risk interpretation
- Top SHAP factors for that individual prediction
- Links to the project author's GitHub and LinkedIn profiles

GitHub: https://github.com/InfinitePraveen

LinkedIn: https://www.linkedin.com/in/infinitepraveen/

## Interview discussion points

### Why XGBoost?

XGBoost is a strong choice for structured/tabular customer data because boosted decision trees can model nonlinear relationships and interactions without requiring feature scaling.

### Why SHAP?

A churn probability alone does not explain the decision. SHAP attributes the model output to individual features, allowing the project to answer questions such as: *Which customer characteristics pushed this prediction toward churn?*

For tree models, SHAP's TreeExplainer uses Tree SHAP algorithms specifically optimized for tree ensembles.

### Important limitation

SHAP explains the behavior of the trained predictive model. A SHAP contribution should not automatically be interpreted as a causal effect or as proof that changing a feature will change a customer's behavior.

## License and dataset note

The application code in this repository is provided for learning and portfolio use. Please review the original dataset repository's license and terms before redistributing the raw dataset.
