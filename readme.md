# Customer Churn Prediction

Machine learning project to predict whether a telecom customer will churn.

## Approach

The project includes:

- Data cleaning
- Feature engineering
- Handling missing values
- One-hot encoding categorical features
- Feature scaling
- Machine learning pipeline
- Hyperparameter tuning using GridSearchCV

## Model

Final model:

- Gradient Boosting Classifier

The model was selected after comparing multiple classification algorithms and tuning hyperparameters.

## Evaluation Metrics

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Deployment Preparation

- Saved trained pipeline using Joblib
- Created prediction script for new customer predictions

## Libraries Used

- Pandas
- NumPy
- Scikit-learn
- Joblib