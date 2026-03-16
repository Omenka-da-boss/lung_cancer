# python -m scripts/run_pipeline 
import pandas as pd
import numpy as np


def preprocess(df: pd.DataFrame,target_col:str = "lung_cancer") -> pd.DataFrame:
    
    df.columns = df.columns.str.strip()
    
    for col in ["patient_id","patient_ID"] :
        if col in df.columns:
            df = df.drop([col],axis=1)
    
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].map({"Yes":1,"No":0})
    
    nulls = df.isnull().sum()
    
    wow = dict(nulls)
    
    for c,d in wow.items():
        if df[c].dtype == "object":
            if d > 0:
                df[c] = df[c].fillna("Unknown")
    
    num_cols = [c for c in df.select_dtypes(["int64","float64"]).columns]
    
    for c in num_cols:
        df[c] = df[c].fillna(0)
    
    return df
