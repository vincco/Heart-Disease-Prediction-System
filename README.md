# Heart Disease Prediction System

A machine-learning project that predicts a binary heart-disease outcome
from clinical features. It includes data preprocessing, model
comparison, evaluation reports, patient-level predictions, a Streamlit
dashboard, and a FastAPI REST API.

> **Disclaimer:** This is an educational project, not a medical device.
> Predictions are not a diagnosis and must not replace advice from a
> qualified healthcare professional. Do not use it for clinical
> decisions.

## Contents

-   [Overview](#overview)
-   [Features](#features)
-   [Technology stack](#technology-stack)
-   [Project structure](#project-structure)
-   [Machine-learning workflow](#machine-learning-workflow)
-   [Algorithms](#algorithms)
-   [Installation](#installation)
-   [Run the dashboard](#run-the-dashboard)
-   [Run the REST API](#run-the-rest-api)
-   [Make predictions](#make-predictions)
-   [Train the models](#train-the-models)
-   [Evaluation](#evaluation)
-   [Limitations](#limitations)

## Overview

The application accepts clinical measurements and returns a
model-estimated probability and binary classification. The training
workflow compares multiple classifiers and saves a fitted pipeline so
that inference can reuse the same preprocessing steps used during
training.

The supplied project uses a heart-disease dataset with 13 input features
and a binary target. Data-cleaning steps should be reviewed before
retraining, especially target-label orientation and codes that represent
missing values.

## Features

-   Data auditing and preprocessing
-   Numerical imputation and feature scaling
-   Categorical one-hot encoding
-   Comparison of several supervised classifiers
-   Cross-validation and hyperparameter search
-   Classification metrics and visualizations
-   Probability threshold selection
-   Permutation feature importance and per-patient sensitivity
    explanations
-   K-means clustering and PCA visualization
-   Interactive Streamlit interface
-   FastAPI prediction endpoints
-   Single-record and batch prediction workflows
-   Saved model artifacts and training report

## Technology stack

-   Python
-   pandas and NumPy
-   scikit-learn
-   joblib
-   Streamlit
-   FastAPI and Pydantic
-   Uvicorn
-   Plotly

Exact dependencies and versions are defined in `requirements.txt`.

## Project structure

The following is the expected layout based on the project modules. Some
generated files may not be present until training has been run.

``` text
heart_disease_prediction/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── data/
│   └── heart.csv                  # Input dataset
├── src/
│   ├── config.py                  # Configuration and feature definitions
│   ├── preprocess.py              # Data auditing and preprocessing
│   ├── train.py                   # Model training, selection and evaluation
│   ├── predict.py                 # Single and batch prediction functions
│   ├── api.py                     # FastAPI REST service
│   ├── charts.py                  # Visualizations
│   └── insights.py                # Dataset insights
└── model/
    ├── heart_disease_pipeline.joblib  # Saved classifier pipeline
    ├── cluster_model.joblib           # Saved clustering artifact
    └── training_report.json           # Evaluation report
```

## Machine-learning workflow

1.  **Load data:** Read the clinical dataset from `data/heart.csv`.
2.  **Audit and clean:** Check duplicates, invalid/missing-value codes,
    and target-label conventions.
3.  **Preprocess features:** Impute missing values, scale numerical
    variables, and one-hot encode categorical variables.
4.  **Split data:** Create a stratified training/test split.
5.  **Train and tune:** Fit candidate models and tune hyperparameters
    using cross-validation.
6.  **Evaluate:** Compare models using metrics such as ROC-AUC,
    accuracy, precision, recall, F1, and Brier score where available.
7.  **Choose a threshold:** Use training-derived out-of-fold predictions
    to select a classification threshold.
8.  **Save artifacts:** Serialize the fitted preprocessing/model
    pipeline and associated metadata.
9.  **Serve predictions:** Use the same saved pipeline in the Streamlit
    app or FastAPI service.
10. **Explore clusters:** Use K-means and PCA to explore groups in the
    feature data. Clusters are exploratory and are not diagnoses.

Keeping preprocessing inside the saved pipeline helps reduce
training-serving skew: prediction-time inputs go through the same fitted
transformations as training data.

## Algorithms

The model-comparison workflow includes the following candidate
classifiers:

-   Logistic Regression
-   K-Nearest Neighbours (KNN)
-   Support Vector Machine (SVM)
-   Gaussian Naive Bayes
-   Decision Tree
-   Random Forest
-   Gradient Boosting
-   Soft-voting ensemble

The project also uses:

-   **StandardScaler** for numerical feature standardization
-   **Imputation** for missing values
-   **OneHotEncoder** for categorical features
-   **Stratified train/test splitting**
-   **Repeated stratified cross-validation** and hyperparameter search
-   **ROC-AUC** and other classification metrics for model evaluation
-   **Permutation importance** for global feature-importance analysis
-   **K-means** for unsupervised clustering
-   **PCA** for two-dimensional visualization

Not every algorithm discussed in the associated course handout is
implemented here. For example, XGBoost, LightGBM, DBSCAN, UMAP, t-SNE,
reinforcement learning, and automated production drift monitoring are
not part of the core workflow.

## Installation

### 1. Extract the project

Extract the ZIP archive and open a terminal in the project root (the
folder containing `app.py` and `requirements.txt`).

### 2. Create a virtual environment

macOS/Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

``` powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a command fails, check that the terminal is in the project root and
that the virtual environment is activated.

## Run the dashboard

From the project root:

``` bash
streamlit run app.py
```

Streamlit prints a local URL, commonly `http://localhost:8501`. Open
that URL in a browser.

## Run the REST API

Open a second terminal, activate the same virtual environment, and
navigate to the project root. Start the API with:

``` bash
uvicorn src.api:app --reload
```

The API is usually available at `http://127.0.0.1:8000`.

-   Health check: `GET /health`
-   Single prediction: `POST /predict`
-   Interactive API documentation: `http://127.0.0.1:8000/docs`

Use the exact request schema displayed in `/docs`; the API validates the
expected patient fields and allowed values. If the model artifacts are
missing, train the project first.

## Make predictions

### Through the dashboard

1.  Start Streamlit.
2.  Open the local dashboard URL.
3.  Enter the required patient features.
4.  Submit the form to see the model's prediction and probability.
5.  If supported by the selected page, upload a CSV for batch
    predictions and download the results.

### Through the API

Open `/docs`, expand `POST /predict`, select **Try it out**, enter a
JSON body matching the `Patient` schema, and execute the request.

An optional `explain=true` query parameter may be available for a
feature-sensitivity explanation. Refer to the generated API
documentation for the precise parameters and response structure.

## Train the models

From the project root, with the virtual environment active:

``` bash
python src/train.py
```

Training can take longer than starting the dashboard because
cross-validation and hyperparameter searches fit models repeatedly. The
script should generate or update the model artifacts and evaluation
report.

Do not retrain merely to launch the existing dashboard if the supplied
artifacts are already present and compatible with your installed library
versions.

## Evaluation

Common classification metrics used in the project include:

-   **Accuracy:** proportion of correct predictions
-   **Precision:** proportion of predicted positive cases that are truly
    positive
-   **Recall/sensitivity:** proportion of actual positive cases detected
-   **Specificity:** proportion of actual negative cases correctly
    identified
-   **F1-score:** harmonic mean of precision and recall
-   **ROC-AUC:** ranking performance across classification thresholds
-   **Brier score:** squared-error measure for predicted probabilities

A probability is not automatically calibrated just because it is between
zero and one. Model performance and probability reliability can change
on new populations, so the saved evaluation report should be treated as
an estimate rather than a guarantee.

## Limitations

-   The dataset is small and may not represent other populations or
    clinical settings.
-   Evaluation results do not establish clinical safety or usefulness.
-   A predicted probability is not a diagnosis.
-   Feature sensitivity explanations are not causal explanations.
-   K-means clusters are exploratory groups, not medical categories.
-   The project does not, by itself, provide a complete production
    monitoring system, automatic drift-triggered retraining, or a
    managed model registry.
-   Serialized `joblib`/pickle artifacts should only be loaded from
    trusted sources.

## Learning objectives

This project demonstrates an end-to-end classical ML workflow: data
preparation, feature preprocessing, supervised classification,
unsupervised clustering, model evaluation, artifact packaging, and
serving predictions through a web UI and REST API.
