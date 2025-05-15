from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.models import HouseFeatures
from models.models import HouseFeatures
import pickle
import os
import numpy as np

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# For demonstration, we use the same features as in house_prediction.py
FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'house_age',
    'was_renovated', 'total_sqft'
]

app = FastAPI(title="House Price Prediction API")

@app.get("/")
def root():
    return {"message": "Welcome to the House Price Prediction API!"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        X = np.array([[getattr(features, feat) for feat in FEATURES]])
        price = model.predict(X)[0]
        return {"predicted_price": price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/features")
def get_features():
    return {"features": FEATURES}

# Optionally, add a batch prediction endpoint
class BatchRequest(BaseModel):
    houses: list[HouseFeatures]

@app.post("/batch_predict")
def batch_predict(batch: BatchRequest):
    try:
        X = np.array([[getattr(h, feat) for feat in FEATURES] for h in batch.houses])
        prices = model.predict(X).tolist()
        return {"predicted_prices": prices}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
