import os
from fastapi import FastAPI,Request
from fastapi.responses import Response
from pydantic import BaseModel
import sys
import pandas as pd
import uvicorn
import time
from collections import deque
# from prometheus_client import Counter,Histogram,Guage,generate_latest(),CONTENT_TYPE_LATEST
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.middleware.cors import CORSMiddleware

# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from .inference import predict

app = FastAPI()

# PROMETHEUS METRICS
# counter for the request made to our api
REQUESTS = Counter("http_requests_total","Total HTTP requests",["method","endpoint","status"])

# counter for errors in the api
ERRORS = Counter("http_errors_total","Total HTTP errors",["method","endpoint"])

# counter for good prediction
PREDICTIONS = Counter("model_prediction_total","Total Model prediction",["method","endpoint","status"])

# request latency in seconds
LATENCY = Histogram("http_requests_duration_duration","HTTP requests latency",labelnames=['method', 'endpoint'],buckets=(0.05,0.1,0.25,0.5,1.0,2.0,5.0))

# 
PREDICTION_VALUE = Histogram("model_prediction_value","Prediction output values")

LATEST_PREDICTION = Gauge("model_last_prediction_value","Last prediction value")

HEALTH_CHECK = Gauge("model_health_score","Average Score for last 10 predictions")

prediction_history = deque(maxlen=10)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def monitor_api(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    status_code = response.status_code
    
    REQUESTS.labels(method=method,endpoint=endpoint,status=status_code).inc()
    LATENCY.labels(method=method,endpoint=endpoint).observe(duration)
    
    if status_code == 200:
        PREDICTIONS.labels(method=method,endpoint=endpoint,status=status_code).inc()
    
    if status_code >= 400:
        ERRORS.labels(method=method,endpoint=endpoint,status=status_code).inc()
    
    return response
    
        
    

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
        out,nums = predict(data.dict())
        
        new_df = pd.DataFrame([data.dict()])
        new_df.to_csv("monitoring\\new_data.csv",mode="a",header=False,index=False)
        
        prediction_history.append(out)
        avg = sum(prediction_history) / len(prediction_history) if prediction_history else 0.0
        HEALTH_CHECK.set(avg)
        PREDICTION_VALUE.observe(nums)
        
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/metrics")
def get_metrics():
    return Response(content=generate_latest(),media_type=CONTENT_TYPE_LATEST)
@app.get("/health")
def get_health():
    if not prediction_history:
        return {"status":"Healthy","last_10_avg": 0.0,"sample_used":0}
    avg = sum(prediction_history)/len(prediction_history)
    
    health_threshold = 0.8
    status = ("Healthy" if avg > health_threshold else "Unhealthy")
    return {
        "status": status,
        "last_10_avg": avg,
        "sample_used": len(prediction_history)
    }
    


if __name__ == "__main__":
    uvicorn.run(app, host="16.171.224.225", port=8000)