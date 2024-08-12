from typing import List
import pandas as pd
from fastapi import FastAPI, Body, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
import holidays
import pickle
import os

app = FastAPI()

# Define the request and response models
class PredictRequest(BaseModel):
    trip_id: str
    tpep_pickup_datetime: str
    trip_distance: float
    PULocationID: int
    DOLocationID: int
    Airport: bool


class PredictResponse(BaseModel):
    prediction: List[float]

# Load the model
def get_model():
    model_path = "../../models/random_forest_regressor_model.joblib"  # Adjust path as needed
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found at '{model_path}'")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def check_holiday_weekend(date):
    ny_holidays = holidays.US(state='NY')
    is_weekend = date.weekday() >= 5
    is_holiday = date in ny_holidays
    return pd.Series([is_weekend, is_holiday])

def preprocess_request(data: List[PredictRequest]):
    # Convert the list of PredictRequest to DataFrame
    df = pd.DataFrame([{
        "tpep_pickup_datetime": item.request_datetime,  # Use request_datetime as pickup time
        "pickup_longitude": None,  # Not provided, set placeholders
        "pickup_latitude": None,  # Not provided, set placeholders
        "dropoff_longitude": None,  # Not provided, set placeholders
        "dropoff_latitude": None,  # Not provided, set placeholders
        "passenger_count": item.passenger_count,
        "trip_distance": item.trip_distance,
        "PULocationID": item.PULocationID,
        "DOLocationID": item.DOLocationID,
        "Airport": item.Airport
    } for item in data])

    # Convert pickup datetime to pandas datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    # Calculate dummy values; adjust if necessary
    df['congestion_surcharge_dummy'] = 0
    df['airport_fee_dummy'] = df['Airport'].astype(int)
    
    # Add weekend and holiday indicators
    df[['is_weekend', 'is_holiday']] = df['tpep_pickup_datetime'].apply(check_holiday_weekend)
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['is_holiday'] = df['is_holiday'].astype(int)

    # Feature columns
    feature_columns = ['trip_distance', 'is_weekend', 'is_holiday', 
                       'congestion_surcharge_dummy', 'airport_fee_dummy']
    input_df = df[feature_columns]

    return input_df

@app.post("/predict", response_model=PredictResponse)
def predict(
    data: List[PredictRequest] = Body(...),
    model = Depends(get_model)
) -> PredictResponse:
    try:
        # Preprocess the request data
        input_df = preprocess_request(data)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        return PredictResponse(prediction=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
