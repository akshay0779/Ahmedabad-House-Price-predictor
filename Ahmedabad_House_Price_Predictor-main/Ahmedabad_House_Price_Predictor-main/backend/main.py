
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

class HouseFeatures(BaseModel):
    total_sqft: float
    bhk: int
    location: str
    floor_num: float
    price_sqft: float
    
    # Add more fields if your dataset has them


app = FastAPI(
    title="Ahmedabad House Price Predictor",
    version="1.0",
    description="Predict house prices in Ahmedabad using a trained XGBoost model"
)


@app.get("/")
def root():
    return {
        "message": "Welcome to Ahmedabad House Price Predictor API!",
        "endpoints": ["/predict", "/health"]
    }


MODEL_PATH = os.path.join(os.getcwd(), "models", "house_price_pipeline_prefect.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)
print("✅ Pipeline loaded successfully!")


@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API is running"}


@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([features.dict()])

        # Predict
        pred = pipeline.predict(input_df)
        predicted_price = float(pred[0])

        return {"predicted_price_lakhs": predicted_price}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
