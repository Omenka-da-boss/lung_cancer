import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,accuracy_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.load.load import load_data
from src.load.preprocess import preprocess
from src.feature.build_feat import build_feature
from src.utils.validate import validate_data
from src.model.train import train_data
from src.model.test import evaluate_model
from src.model.tune import hyper_tuning

path = './EDA/lung_cancer_dataset.csv'

def main(args):
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    mlpaths = args.mlflow_uri or f"file:\\{project_root}\\mlruns"
    ml_uri = "http://localhost:5000"
    
    mlflow.set_tracking_uri(ml_uri)
    mlflow.set_experiment(args.experiment)
    
    with mlflow.start_run():
        
        mlflow.log_param("Model","xgboost")
        mlflow.log_param("Threshold",args.threshold)
        mlflow.log_param("Test Size",args.test_size)
        
        
        print(f"Loading Of Dataset!!")
        
        df = load_data(args.input)
        
        print(f"✅✅ Dataset Loaded Succcessfully!! with Row: {df.shape[0]} and Columns: {df.shape[1]}")
        
        
        # Validating Data
        
        print("Validating Dataset")
        
        is_valid,failed = validate_data(df)
        
        mlflow.log_metric("Data Quality Passed",int(is_valid))
        
        if failed:
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")
        
        # preprocessing dataset
        
        print("⚒️⚒️ Preprocess Dataset")
        df = preprocess(df)
        print("✅✅ Successfully preprocessed Dataset")
        
        # saving cleaned dataset
        print("Saving of preprocessed dataset!!!")
        processed_path = os.path.join(project_root, "data", "processed", "lung_cancer_dataset_preprocessed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")
        mlflow.log_artifacts(processed_path)
        
        # Feature Engineering Stage
        print("⚒️⚒️ Feature Engineering!!")
        
        print("🛠️  Building features...")
        
        target = args.target
        
        if target not in df.columns:
            raise ValueError(f"Target Column of value {target} not found in dataset columns")
        
        df_eng = build_feature(df,target_cols=target)
        
        for c in df_eng.select_dtypes(include="bool").columns:
            df[c] = df[c].astype(int)
        print(f"✅ Feature engineering completed: {df_eng.shape[1]} features")
        
        artifact_path = os.path.join(project_root,"artifacts")
        os.makedirs(artifact_path,exist_ok=True)
        
        feature_cols = [c for c in df.drop([target],axis=1).columns]
        
        import json
        import joblib
        
        with open(os.path.join(artifact_path,"features_columns.json"),"w")as f:
            json.dump(feature_cols,f)
        
        with open(os.path.join(artifact_path,"features_columns.txt"),"w")as f:
            json.dump(feature_cols,f)
            
        # log the features in mlflow
        mlflow.log_text("\n".join(feature_cols),artifact_file="features_column.txt")
        
        # Log the features for serving the project
        
        preprocessed_columns = {
            "feature_column":feature_cols,
            "target": target
        }
        
        joblib.dump(preprocessed_columns,os.path.join(artifact_path,"preprocessing.pkl"))
        # mlflow.log_artifact(os.path.join(artifact_path,"preprocessing.txt")) 
        mlflow.log_artifact(os.path.join(artifact_path, "preprocessing.pkl"))   
        print(f"✅ Saved {len(feature_cols)} feature columns for serving consistency")
        
        mlflow.log_artifacts(artifact_path)
        
        # split data
        x = df_eng.drop([target],axis=1)
        y = df_eng[target]
        
        # split for training 
        x_train,x_test,y_train,y_test = train_test_split(
            x,y,test_size=args.test_size,random_state=42
        )
        
        # Train Model
        model = train_data(x_train,x_test,y_train,y_test,args.threshold)
        
        # evaluate data
        print("Metrics Of The Default Model Training")
        
        evaluation = evaluate_model(model,x_test,y_test,args.threshold)
        
        
        # Tuning Of Data
        tune_params = hyper_tuning(x_train,y_train,x_test,y_test,args.threshold,y)
        
        # Training with tuning params 
        
        
        mlflow.log_params(tune_params)
    
        # Training Time
        start_time = time.time()
        new_model = CatBoostClassifier(**tune_params)
        new_model.fit(x_train,y_train)
        end_time = time.time() - start_time
        mlflow.log_metric("Train Time",end_time)
        
        # Prediction Time
        start_time = time.time()
        preds = new_model.predict_proba(x_test)[:,1]
        y_preds = (preds >= args.threshold).astype(int)
        pred_time = time.time() - start_time
        mlflow.log_metric("Prediction Time",pred_time)
        
        # Metrices
        precision = precision_score(y_test,y_preds,pos_label=1)
        recall = recall_score(y_test,y_preds,pos_label=1)
        f1 = f1_score(y_test,y_preds,pos_label=1)
        roc_auc = roc_auc_score(y_test,y_preds)
        acc = accuracy_score(y_test,y_preds)
        
        # Log Metrics
        mlflow.log_metric("Precision",precision)
        mlflow.log_metric("Recall",recall)
        mlflow.log_metric("F1 Score",f1)
        mlflow.log_metric("Roc_Auc Score",roc_auc)
        mlflow.log_metric("Accuracy Score",acc)
        
        from mlflow.models import infer_signature
        
        # saving new model
        
        signature = infer_signature(x_train,model.predict(x_train))
        mlflow.sklearn.log_model(
            new_model, 
            signature=signature,
            artifact_path="model",  # This creates a 'model/' folder in MLflow run artifacts
            registered_model_name="Updated Project"
        )
        
        # Save model using pickle or job lib
        
        print("Saving Model To Folder")
        joblib.dump(new_model,"models\\model.pkl")
        print("Model Successfully Loaded ✅✅✅")
        
        
        train_ds = mlflow.data.from_pandas(df,source="training_data")
        mlflow.log_input(train_ds,context="Training")
        
        # === Final Performance Summary ===
        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time: {start_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        # print(f"   Samples per second: {len(x_test)/pred_time:.0f}")
        print(f"   Accuracy Score: {acc * 100 :.0f}%")
        
        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_preds, digits=3))
        
    

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="lung_cancer")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="LungCancerExperiment")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args =p.parse_args() 
    
    main(args)        
            
            
"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \                                            
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn

"""
# python scripts/test_data.py