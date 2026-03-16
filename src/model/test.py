from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import mlflow



def evaluate_model(model,x_test,y_test,threshold):
    
    preds = model.predict_proba(x_test)[:, 1]
    pred = (preds >= threshold).astype(int)
    acc = accuracy_score(y_test,pred)
    roc_auc = roc_auc_score(y_test,pred)
    pre = precision_score(y_test,pred, pos_label=1)
    recall = recall_score(y_test,pred, pos_label=1)
    f1 = f1_score(y_test,pred, pos_label=1)
    
    clas = classification_report(y_test,pred)
    cons = confusion_matrix(y_test,pred)
    
    mlflow.log_metric("Precision",pre)
    mlflow.log_metric("Recall",recall)
    mlflow.log_metric("F1 Score",f1)
    mlflow.log_metric("Roc_Auc Score",roc_auc)
    mlflow.log_metric("Accuracy Score",acc)
    
    print(f"\n Classification Report \n {clas} \n Confusion Matrix \n {cons}")
    print(f"\n Precision Score \n {pre} \n Recall Score \n {recall}")
    print(f"\n Accuracy Score \n {acc} \n F1 Score \n {f1}")
    