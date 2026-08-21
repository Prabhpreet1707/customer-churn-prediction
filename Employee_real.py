import pandas as pd

df = pd.read_csv("Employee_real.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["TotalSpend"] = (df["MonthlyCharges"] * df["tenure"])

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import(
    confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
)

df = df.drop("customerID", axis=1)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Automatically detect columns
numerical_columns = X.select_dtypes(exclude="object").columns

categorical_columns = X.select_dtypes(include="object").columns

numerical_transformation = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

category_transformation = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

column_transformation = ColumnTransformer([
    ("num", numerical_transformation, numerical_columns),
    ("cat", category_transformation, categorical_columns)
])

pipeline = Pipeline([
    ("column" , column_transformation),
    ("model", GradientBoostingClassifier( random_state=42, n_estimators=100,learning_rate=0.1))
])

param = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [3, 5, 10],
    "model__learning_rate": [0.01, 0.1, 0.2]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param,
    cv=5,
    scoring="recall",
    n_jobs=-1
)

grid.fit(X_train, y_train)

predictions = grid.predict(X_test)
print(grid.best_params_)

print()

print(grid.best_score_)


print(confusion_matrix(y_test, predictions))

print(accuracy_score(y_test, predictions))
print(precision_score(y_test, predictions))
print(recall_score(y_test, predictions))
print(f1_score(y_test, predictions))


probabilities = grid.predict_proba(X_test)[:, 1]  # probability of "will churn" for each customer

custom_threshold = 0.3  # lower than default 0.5 — catches more churners, at the cost of more false positives
custom_predictions = (probabilities >= custom_threshold).astype(int)

print(confusion_matrix(y_test, custom_predictions))
print("Recall:", recall_score(y_test, custom_predictions))
print("Precision:", precision_score(y_test, custom_predictions))

import joblib

joblib.dump(grid.best_estimator_, "customer_churn_model.pkl")

print()

print("Model saved successfully!")