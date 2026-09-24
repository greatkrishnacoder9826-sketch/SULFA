# 💳 Credit Risk Prediction System

A machine learning project for predicting **credit/loan default risk** using applicant financial and demographic information.

The project focuses on building a complete and interpretable classification pipeline using **Logistic Regression and XGBoost**, with cross-validation, hyperparameter tuning, probability calibration, SHAP explainability, and error analysis.

> 🚧 **Project Status:** Machine Learning pipeline completed.
> 🔜 **Next Step:** FastAPI backend + frontend interface.

---

## 📌 Project Overview

Credit risk prediction is a classification problem where the goal is to estimate whether a loan applicant is likely to **default on a loan**.

This project takes applicant information such as:

* Age
* Income
* Employment length
* Home ownership
* Loan intent
* Loan grade
* Loan amount
* Interest rate
* Loan-to-income ratio
* Credit history length
* Previous default information

and uses machine learning to predict the applicant's loan status.

---

## 🎯 Objectives

The main objectives of this project are:

* Perform Exploratory Data Analysis (EDA)
* Clean and preprocess raw data
* Handle missing values
* Handle class imbalance
* Build preprocessing pipelines
* Train multiple classification models
* Compare model performance using cross-validation
* Tune XGBoost hyperparameters
* Optimize the classification threshold
* Calibrate predicted probabilities
* Explain model predictions using SHAP
* Analyze false positives and false negatives
* Save the final trained model for deployment

---

## 🧠 Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Data Cleaning
     │
     ▼
Feature / Target Separation
     │
     ▼
Train-Test Split
     │
     ▼
Class Imbalance Handling
     │
     ▼
Data Preprocessing
 ┌───────────────┐
 │ Numerical     │──► Imputation
 │ Features      │    Scaling (LR)
 └───────────────┘
 │
 │
 ┌───────────────┐
 │ Categorical   │──► Imputation
 │ Features      │    One-Hot Encoding
 └───────────────┘
     │
     ▼
Model Training
 ┌──────────────────────┐
 │ Logistic Regression  │
 │ XGBoost              │
 └──────────────────────┘
     │
     ▼
Stratified Cross Validation
     │
     ▼
XGBoost Hyperparameter Tuning
     │
     ▼
Threshold Optimization
     │
     ▼
Probability Calibration
     │
     ▼
SHAP Explainability
     │
     ▼
Error Analysis
     │
     ▼
Saved Model (.pkl)
     │
     ▼
FastAPI Backend
     │
     ▼
Frontend Interface
```

---

## 🔍 Exploratory Data Analysis

The project performs several EDA steps including:

* Dataset shape and information
* Missing-value analysis
* Target class distribution
* Numerical feature analysis
* Outlier detection using box plots
* Descriptive statistics
* Correlation analysis

The dataset is also checked for unrealistic values such as:

* Age below 18
* Age above 100
* Excessive employment length
* Invalid loan amounts
* Duplicate records

---

## 🧹 Data Preprocessing

Different preprocessing strategies are applied to numerical and categorical features using `ColumnTransformer`.

### Numerical Features

```text
person_age
person_income
person_emp_length
loan_amnt
loan_int_rate
loan_percent_income
cb_person_cred_hist_length
```

For Logistic Regression:

```text
Missing values → Median Imputation → Standard Scaling
```

For XGBoost:

```text
Missing values → Median Imputation
```

### Categorical Features

```text
person_home_ownership
loan_intent
loan_grade
cb_person_default_on_file
```

Processing:

```text
Missing values → Constant Imputation → One-Hot Encoding
```

`handle_unknown="ignore"` is used in `OneHotEncoder` so that unseen categories during prediction do not cause errors.

---

## ⚖️ Handling Class Imbalance

The target variable contains an imbalance between the classes.

The project calculates a class weight using:

```python
neg, pos = np.bincount(y_train)
scale_weights = neg / pos
```

This value is then used to reduce the impact of class imbalance during model training.

---

## 🤖 Models

### 1. Logistic Regression

A Logistic Regression model is used as the baseline model.

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

### 2. XGBoost

XGBoost is used as the main tree-based classification model.

The project also performs hyperparameter tuning using `RandomizedSearchCV`.

Parameters explored include:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`

---

## 🔄 Stratified Cross-Validation

A 5-fold `StratifiedKFold` cross-validation strategy is used.

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The following metrics are evaluated:

* ROC-AUC
* Accuracy
* Precision
* Recall
* F1 Score

Stratification helps maintain a similar class distribution across the folds.

---

## 🎯 Threshold Optimization

Instead of relying only on the default classification threshold, the project analyzes different probability thresholds.

Precision-Recall values are calculated and the threshold producing the highest F1 score is selected.

This allows the model's classification behavior to be adjusted according to the desired precision/recall trade-off.

---

## 📊 Probability Calibration

The project also performs probability calibration using:

```python
CalibratedClassifierCV(
    best_model,
    method="sigmoid",
    cv=5
)
```

The calibrated and uncalibrated probabilities are compared using a calibration curve.

This is particularly useful when the model's output probability is intended to represent an estimate of risk.

---

## 🔎 Model Explainability with SHAP

SHAP is used to understand how individual features influence model predictions.

The project includes:

### Global Explanation

A SHAP summary plot is generated to understand which features have the largest overall influence on predictions.

### Local Explanation

A SHAP waterfall plot is generated for an individual applicant to show how different features contributed to that specific prediction.

---

## 🚨 Error Analysis

The project separately analyzes:

### False Positives

Applicants predicted as default but actually belonging to the non-default class.

### False Negatives

Applicants predicted as non-default but actually belonging to the default class.

This helps understand where the model is making mistakes instead of relying only on aggregate performance metrics.

---

## 💾 Model Saving

The calibrated model is saved using `joblib`:

```python
joblib.dump(
    calibrated_model,
    "credit_risk_model.pkl"
)
```

The saved model will later be loaded by the FastAPI backend to generate predictions.

---

## 🛠️ Tech Stack

| Technology       | Purpose                      |
| ---------------- | ---------------------------- |
| Python           | Programming language         |
| Pandas           | Data manipulation            |
| NumPy            | Numerical computation        |
| Matplotlib       | Data visualization           |
| Seaborn          | EDA visualization            |
| Scikit-learn     | Preprocessing & ML           |
| XGBoost          | Gradient boosting model      |
| SciPy            | Hyperparameter distributions |
| SHAP             | Model explainability         |
| Joblib           | Model serialization          |
| Jupyter Notebook | Model development            |

### Planned

| Technology          | Purpose                    |
| ------------------- | -------------------------- |
| FastAPI             | Backend API                |
| HTML/CSS/JavaScript | Frontend                   |
| REST API            | Model prediction interface |

---

## 📂 Project Structure

The project is currently being developed as a notebook-based ML project.

The planned deployment structure is:

```text
credit-risk-prediction/
│
├── data/
│   └── credit_risk_dataset.csv
│
├── notebooks/
│   └── credit_risk_analysis.ipynb
│
├── model/
│   └── credit_risk_model.pkl
│
├── backend/
│   ├── main.py
│   ├── schemas.py
│   └── model_service.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The `backend/` and `frontend/` components are planned for the next phase of the project.

---

## 🚀 Future Development

The next stage of this project will convert the trained machine learning model into a complete web application.

### Phase 1 — FastAPI Backend

The FastAPI backend will:

1. Load the trained `.pkl` model
2. Accept applicant information through an API
3. Validate input data
4. Generate prediction probabilities
5. Return the predicted risk and probability

Example planned endpoint:

```text
POST /predict
```

### Phase 2 — Frontend

A frontend interface will be created where users can enter applicant information through a form.

The frontend will communicate with the FastAPI backend and display the prediction result.

### Phase 3 — Deployment

The complete application can later be deployed as a web application.

---

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes**.

A real-world credit decision system would require extensive validation, appropriate financial-domain controls, privacy protection, fairness/bias evaluation, regulatory compliance, and monitoring before being used for actual lending decisions.

---

## 👨‍💻 Author

**Krishna Great**

Focused on:

* Data Science
* Machine Learning
* Generative AI
* Agentic AI
* Backend Development

---

## ⭐ Project Roadmap

```text
[x] Data Cleaning
[x] Exploratory Data Analysis
[x] Feature Engineering / Selection
[x] Preprocessing Pipelines
[x] Logistic Regression
[x] XGBoost
[x] Stratified Cross-Validation
[x] Hyperparameter Tuning
[x] Threshold Optimization
[x] Probability Calibration
[x] SHAP Explainability
[x] Error Analysis
[x] Model Serialization
[ ] FastAPI Backend
[ ] Frontend
[ ] API Integration
[ ] Deployment
```

---

### 📌 Current Status

**Machine Learning Pipeline:** ✅ Completed
**Model Serialization:** ✅ Completed
**FastAPI Backend:** 🔜 Planned
**Frontend:** 🔜 Planned
**Deployment:** 🔜 Planned
