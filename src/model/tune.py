import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def hyper_tuning(x_train,y_train,x_test,y_test,threshold,y):
    
    
    def objective(trial):
        param = {
            "iterations": trial.suggest_int("iterations", 100, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "auto_class_weights":'Balanced',
            "verbose": False,
            "allow_writing_files": False
    }
    

        model = CatBoostClassifier(**param)
        # model = XGBClassifier(**params)

        model.fit(x_train, y_train,eval_set=[(x_test,y_test)])

        
        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.35).astype(int)
        # preds = model.predict(x_test)
        acc = accuracy_score(y_test,preds) * 100
        acc = round(acc,2)
        
        if acc >= 71.5:
            print("Accuracy Score:",acc)
            # return recall_score(y_test, preds)
            return recall_score(y_test,preds)
        else: 
            return "Too Low Accuracy"


    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=30)
    
    params = study.best_params
    
    model = XGBClassifier(**params)
    
    return model,study.best_params



