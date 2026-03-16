import os
from fastapi import FastAPI
from pydantic import BaseModel
import sys


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.serving.inference import predict

app = FastAPI()

@app.get("/")
def status():
    return {"Status":"OK"}


class LungCancer(BaseModel):
    age : int
    gender : str
    pack_years : float
    radon_exposure : str
    asbestos_exposure : str
    secondhand_smoke_exposure : str
    copd_diagnosis : str
    alcohol_consumption : str
    family_history : str


@app.post("/predict")
def predict(data:LungCancer):
    try:
        preds = predict(data.dict())
        return {"Prediction: ",preds}
    except Exception as e:
        return {"Error": {e}}