import os
import sys
import pytest

# Add the app directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '12factorapp')))

# --- Tests for api.py ---
from fastapi.testclient import TestClient
from 12factorapp.api import app, FEATURES  # changed import

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_predict_endpoint():
    # Use dummy data with correct feature names
    payload = {feat: 1 for feat in FEATURES}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()

# --- Tests for house_prediction.py ---
import numpy as np
import pandas as pd
from 12factorapp import house_prediction  # changed import

def test_preprocess_data_returns_expected_shapes():
    df = pd.DataFrame({
        'bedrooms': [3], 'bathrooms': [2.0], 'sqft_living': [1500], 'sqft_lot': [5000],
        'floors': [1.0], 'waterfront': [0], 'view': [0], 'condition': [3],
        'yr_built': [2000], 'yr_renovated': [0], 'date': ['2020-01-01'], 'price': [300000]
    })
    # Add required columns for preprocess_data
    df['sale_year'] = 2020
    df['sale_month'] = 1
    X, y = house_prediction.preprocess_data(df)
    assert X.shape[0] == 1
    assert y.shape[0] == 1

def test_model_predicts_with_sample_input():
    # Use the trained model and scaler from house_prediction
    model = house_prediction.model
    scaler = house_prediction.scaler
    X_test = house_prediction.X_test
    if X_test.shape[0] > 0:
        sample = X_test.iloc[0].values.reshape(1, -1)
        sample_scaled = scaler.transform(sample)
        pred = model.predict(sample_scaled)
        assert pred.shape == (1,)
