from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import pickle
import os
import numpy as np

# Use Pydantic BaseSettings for environment variables
class Settings(BaseSettings):
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')
    API_TITLE: str = "House Price Prediction API"

    model_config ={
        "extra":"allow",
        "env_file": os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    }
settings = Settings()

class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    house_age: int
    was_renovated: int
    total_sqft: float

# Load the trained model
MODEL_PATH = settings.MODEL_PATH
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# For demonstration, we use the same features as in house_prediction.py
FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'house_age',
    'was_renovated', 'total_sqft'
]

API_TITLE = settings.API_TITLE
app = FastAPI(title=API_TITLE)

@app.get("/")
def root():
    return {"message": "Welcome to the House Price Prediction API! Goto /docs for the full functionality."}

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
