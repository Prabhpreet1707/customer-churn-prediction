# Customer Churn Prediction Using Gradient Boosting

## Problem Statement

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers directly impacts revenue, so identifying customers who are likely to leave allows companies to take proactive retention actions.

The goal of this project is to build a machine learning classification model that predicts whether a customer is likely to churn based on their demographic details, service usage, contract information, and billing history.

---

## Project Workflow

The project follows a complete machine learning workflow:

1. Data Understanding and Exploratory Data Analysis (EDA)
2. Data Cleaning and Feature Engineering
3. Data Preprocessing
4. Model Training
5. Hyperparameter Optimization
6. Model Evaluation
7. Model Saving and Prediction

---

## Data Analysis and Preprocessing

During Exploratory Data Analysis, several issues were identified in the dataset:

### 1. Data Type Correction

The `TotalCharges` column was incorrectly stored as an object data type instead of a numerical type (`int64`/`float64`).

It was converted into a numerical column using:

- `pd.to_numeric()`
- Missing values were handled using median imputation

---

### 2. Target Variable Transformation

The target variable `Churn` contained categorical values: Yes and No

Since this was our prediction target, it could not be processed using OneHot Encoding.

Instead, it was converted using mapping:
Yes → 1
No → 0


---

### 3. Removing Unnecessary Features

The `customerID` column was removed because it only represents unique customer identifiers and does not provide meaningful information for predicting churn.

---

### 4. Feature Engineering

A new feature was created:
TotalSpend = MonthlyCharges × tenure

This represents the total amount spent by a customer during their subscription period and helps the model understand customer value.

---

## Model Development

The data was split into training and testing sets using stratified splitting to maintain the same churn distribution in both datasets.

A preprocessing pipeline was created:

### Numerical Features

- Missing value handling using mean imputation
- Feature scaling using StandardScaler

### Categorical Features

- Missing value handling using most frequent imputation
- Encoding using OneHotEncoder

A complete Scikit-Learn Pipeline was used to ensure preprocessing steps were applied consistently during training and prediction.

---

## Model Used

The final model used was:

### Gradient Boosting Classifier

Hyperparameter tuning was performed using:

- GridSearchCV
- 5-fold Cross Validation
- Recall as the optimization metric

The reason for prioritizing recall was that identifying potential churn customers is more important than only maximizing overall accuracy.

---

## Threshold Optimization

The default classification threshold of 0.5 was adjusted.

A custom threshold of:
0.3


was selected because it significantly improved the recall score.

This allows the model to identify more potential churn customers, reducing the chances of missing customers who may leave.

---

## Model Deployment Preparation

The trained pipeline was saved using:joblib

The saved model can be loaded and used for predicting churn on new customer data.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Gradient Boosting
- Git/GitHub

---

## Future Improvements

- Build a FastAPI backend for real-time predictions
- Deploy the model using cloud services
- Add monitoring for model performance
- Experiment with advanced models such as XGBoost and LightGBM

---

## Author

Prabhpreet Singh
