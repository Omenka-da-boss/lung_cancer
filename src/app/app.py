import os
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import pandas as pd


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
def api_predict(data: LungCancer):
    try:
        # df = pd.DataFrame([data.dict()])
        out = predict(data.dict())
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}