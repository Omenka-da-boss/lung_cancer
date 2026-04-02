from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from mlflow import xgboost
import pandas as pd
import mlflow
import time
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier


def train_data(x_train,x_test,y_train,y_test,threshold):
    
    # # split data
    # x = df.drop([target_col],axis=1)
    # y = df[target_col]
    
    # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
  

    model = CatBoostClassifier(iterations=1000)
    # model = XGBClassifier()
    
        # Training Time
    start_time = time.time()
    model.fit(x_train,y_train)
    end_time = time.time() - start_time
    # mlflow.log_metric("Train Time",end_time)
        
        # Prediction Time
    start_time = time.time()
    preds = model.predict_proba(x_test)[:,1]
    y_preds = (preds >= threshold).astype(int)
    pred_time = time.time() - start_time
    # mlflow.log_metric("Prediction Time",pred_time)
        
        # Metrices
    precision = precision_score(y_test,y_preds,pos_label=1)
    recall = recall_score(y_test,y_preds,pos_label=1)
    f1 = f1_score(y_test,y_preds,pos_label=1)
    roc_auc = roc_auc_score(y_test,y_preds)
    acc = accuracy_score(y_test,y_preds)
    
    # acc = accuracy_score(y_test,preds)
    # rec = recall_score(y_test,preds)
    print("Accuracy:",acc)
    print("Recall Score:",recall)
    print("Precsion",precision)
    print("Roc Auc:",roc_auc)
        
        # Log Metrics
    mlflow.log_metric("Precision",precision)
    mlflow.log_metric("Recall",recall)
    mlflow.log_metric("F1 Score",f1)
    mlflow.log_metric("Roc_Auc Score",roc_auc)
    mlflow.log_metric("Accuracy Score",acc)
            
        # Log Model
        
    # mlflow.xgboost.log_model(model,"Model")
        
        # Log training data
        
        
    print(f"Trainied 80% of data and got an accuracy of {acc * 100:.2f} and recall of {recall *100:.2f}")
    return model