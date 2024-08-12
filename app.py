from typing import List
from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime
import yaml

app = FastAPI()

class PredictRequest(BaseModel):
    trip_id: str
    request_datetime: str
    trip_distance: float
    PULocationID: int
    DOLocationID: int
    Airport: int

class PredictResponse(BaseModel):
    prediction: float

def load_model():
    model_path = "models/random_forest_regressor_model.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    return joblib.load(model_path)

with open('./params.yaml', 'r') as file:
    params = yaml.safe_load(file)

def preprocess_request(data: List[PredictRequest]):
    df = pd.DataFrame([{
        "request_datetime": item.request_datetime,
        "trip_distance": item.trip_distance,
        "PULocationID": item.PULocationID,
        "DOLocationID": item.DOLocationID,
        "Airport": item.Airport
    } for item in data])

    df['request_datetime'] = pd.to_datetime(df['request_datetime'])
    df['request_datetime'] = df['request_datetime'].dt.hour

    df['tpep_dropoff_datetime'] = pd.NaT

    df['Airport'] = (df['Airport'] > 0).astype(int)

    feature_columns = [
        'request_datetime',
        'trip_distance',
        'PULocationID',
        'DOLocationID',
        'Airport'
    ]
    input_df = df[feature_columns]
    
    return input_df

@app.post("/predict", response_model=PredictResponse)
def predict(data: List[PredictRequest] = Body(...)):
    try:
        model = load_model()
        input_df = preprocess_request(data)
        predictions = model.predict(input_df)
        return PredictResponse(prediction=predictions[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
