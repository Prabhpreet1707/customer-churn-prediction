import joblib
import pandas as pd

model = joblib.load("customer_churn_model.pkl")

print("Model loaded successfully!")

new_customer = pd.DataFrame({
    "gender": ["Female"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [12],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["Yes"],
    "DeviceProtection": ["Yes"],
    "TechSupport": ["No"],
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["Yes"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [90.5],
    "TotalCharges": [1086],
    "TotalSpend": [1086]
})
prediction = model.predict(new_customer)[0]

probability = model.predict_proba(new_customer)[0][1]


if prediction == 1:
    print("Prediction: Customer will churn")
else:
    print("Prediction: Customer will stay")


print(f"Churn probability: {probability:.2%}")