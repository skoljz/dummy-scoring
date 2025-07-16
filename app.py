from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class CreditApplication(BaseModel):
    age: int
    income: float
    years_employed: int
    children: int

model = joblib.load('model.pkl')
app = FastAPI()

@app.post("/predict")
def predict(applicant: CreditApplication):
    X = [[
        applicant.age,
        applicant.income,
        applicant.years_employed,
        applicant.children
    ]]
    prob = model.predict_proba(X)[0][1]
    approved = prob > 0.5
    return {
        "approved": bool(approved),
        "probability": float(prob)
    }

