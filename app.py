from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("customer_churn_model.pkl")

@app.post("/predict")
def predict(customer: dict):
    df = pd.DataFrame([customer])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }