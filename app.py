from typing import List, Annotated

import numpy as np
import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI()

# Define models and logic for the /predict endpoint
class PredictRequest(BaseModel):
    request_datetime: str
    trip_distance: float
    PULocationID: float
    DOLocationID: float
    Airport: bool


class PredictResponse(BaseModel):
    prediction: float

_model = None

def get_model():
    return None
    global _model
    if _model is None:
        model_path = "catboost_model.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model '{model_path}' not found")
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
    return _model

@app.post("/predict")
def predict(data: Annotated[PredictRequest, Body()], model=Depends(get_model)) -> PredictResponse:
    if not data:
        raise HTTPException(status_code=400, detail="Request body is empty")
    return PredictResponse(prediction=np.random.randn(len(data)))

    # prediction = model.predict(pd.DataFrame(
    #      {"feature1": [o.feature1 for o in data], "feature2": [o.feature2 for o in data]}
    #  ))
    # return PredictResponse(prediction=prediction.tolist())


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()
