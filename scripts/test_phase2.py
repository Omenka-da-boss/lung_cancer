import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna
import os
import sys

print("=== Phase 2: Modeling with XGBoost ===")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from test_data import main


# df = pd.read_csv()

path = "./data/lung_cancer_dataset.csv"

# Make Sure Target column is numeric and 0/1

# if df['lung_cancer'].dtype == "object":
#     df['lung_cancer'] = df['lung_cancer'].str.strip().map({"Yes":1,"No":0})

# assert  df['lung_cancer'].isna().sum() == 0
# assert set(df['lung_cancer'].unique()) <= {0,1}

df = main(path)

 

x = df.drop(['lung_cancer'],axis=1)
y = df['lung_cancer']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

THRESHOLD = 0.4

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    from sklearn.metrics import recall_score
    return recall_score(y_test, y_pred, pos_label=1)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best Params:", study.best_params)
print("Best Recall:", study.best_value)

print("✅✅ Phase 2 Complete.. ")