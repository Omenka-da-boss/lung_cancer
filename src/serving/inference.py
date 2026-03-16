import os
import pandas as pd
import mlflow
from sklearn.preprocessing import LabelEncoder
import joblib

encode = LabelEncoder()

import mlflow.pyfunc
import os
import glob

# MODEL_DIR = "models:/MyModel/1"  # Change this

# try:
#     model = mlflow.pyfunc.load_model(MODEL_DIR)
#     print(f"✅ Model loaded successfully from {MODEL_DIR}")

# except Exception as e:
#     print(f"⚠️ Model not found in registry: {MODEL_DIR}")

#     try:
#         latest_model_paths = glob.glob("./mlruns/*/*/artifacts/model")

#         if latest_model_paths:
#             latest_model = max(latest_model_paths, key=os.path.getmtime)
#             model = mlflow.pyfunc.load_model(latest_model)
#             MODEL_DIR = latest_model
#             print(f"✅ Fallback: Loaded model from {latest_model}")
#         else:
#             raise Exception("No model found in local mlruns")

#     except Exception as fallback_error:
#         raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

MODEL_DIR = "run:7c58635c852547ef90914b0ee8a50e56"

# MODEL_DIR = "/app/model"

try:
    from mlflow.tracking import MlflowClient
    import mlflow.sklearn

    # Connect to MLflow tracking server
    mlflow.set_tracking_uri("http://localhost:5000")

    client = MlflowClient()

    # Find your experiment
    experiment_name = "LungCancerExperiment"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Get the latest run
    runs = client.search_runs(experiment_id, order_by=["start_time DESC"], max_results=1)
    run_id = runs[0].info.run_id

    # Load the model artifact
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Model loaded successfully from {model_uri}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development (OPTIONAL)
    try:
        # Try loading from local MLflow tracking
        import glob
        model_path = './models/model.pkl'
        if model_path:
        #     latest_model = max(local_model_paths, key=os.path.getmtime)
        #     model = mlflow.pyfunc.load_model(latest_model)
            model = joblib.load("./models/model.pkl")
            print(f"✅ Fallback: Loaded model from {model_path}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

try:
    feature_file = os.path.join("artifacts", "features_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
    print(f"✅ Loaded {FEATURE_COLS} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

bin_map = {
    "gender": {"Male":1,"Female":0},
    "asbestos_exposure":{"Yes":1,"No":0},
    "secondhand_smoke_exposure": {"Yes":1,"No":0},
    "family_history": {"Yes":1,"No":0},
    "copd_diagnosis": {"Yes":1,"No":0}
}

multi_map = {
    "radon_exposure" : {"High":2,"Medium":1,"Low":0},
    "alcohol_consumption": {"Heavy":2,"Moderate":1,"Unknown":0}
}

num_cols = ["pack_years","age"]

# def clean_input(df:pd.DataFrame) -> pd.DataFrame:
    
#     """ Transformation Pipeline:
#     1. Clean column names and handle data types
#     2. Apply deterministic binary encoding (using BINARY_MAP)
#     3. One-hot encode remaining categorical features  
#     4. Convert boolean columns to integers
#     5. Align features with training schema and order
#     """
    
#     df = df.copy()
    
#     df.columns = df.columns.str.strip()
    
#     for c in num_cols:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c],errors='coerce')
#             df[c] = df[c].fillna(0)
    
    
#     for c,mapping in bin_map.items():
#         if c in df.columns:
#             df[c] = (df[c].astype(str).str.strip().map(mapping).astype("Int64").fillna(0).astype(int))
    
#     obj_cols = [ c for c in df.select_dtypes(include="object").columns]
    
#     multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
#     for c in  multi_cols:
#         print("Before Encoding ")
#         print(df[c].value_counts())
#         df[c] = df[c].astype(str).str.strip().map(mapping).fillna(0).astype(int)
#         print("After Encoding ")
#         print(df[c].value_counts())
    
#     bool_cols = df.select_dtypes(include=["bool"]).columns
#     if len(bool_cols) > 0:
#         df[bool_cols] = df[bool_cols].astype(int)
#     return df

def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Numeric columns
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # Binary mapping
    for c, mapping in bin_map.items():
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().map(mapping).fillna(0).astype(int)
    
    # Multi-value mapping using multi_map
    for c, mapping in multi_map.items():
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().map(mapping).fillna(0).astype(int)
    
    # Boolean columns (if any)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    return df
    

def predict(input_dict: dict):
    
    df = pd.DataFrame([input_dict])
    
    df_enc = clean_input(df)
    
    try:
        
        preds = model.predict(df_enc)
        
        if hasattr(preds,"tolist"):
            preds = preds.tolist()
        
        if isinstance(preds, (list,tuple)) and len(preds) == 2:
            result = preds[0]
        else:
            result = preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    if result == 1:
        return "Likely to have Lung Cancer"
    else:
        return "Unlikely to have Lung Cancer"
    
"""
try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
    
    if Exception:
        import glob
        
        model_path  =  "models\\model.pkl"
        
        if model_path:
            model = joblib.load(model_path)
        else:
            raise FileExistsError("File not found in directory") 

except Exception as e:
    print(f"✅ Model not found {MODEL_DIR}")
     
    try:
        import glob
        
        model_path  =  "models\\model.pkl"
        
        if model_path:
            model = joblib.load(model_path)
        else:
            raise FileExistsError("File not found in directory")
    
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")
    
"""