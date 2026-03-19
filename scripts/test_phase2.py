import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna
import os
import sys
from catboost import CatBoostClassifier

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
    param = {
            "iterations": trial.suggest_int("iterations", 100, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "verbose": False,
            "allow_writing_files": False
    }

    model = CatBoostClassifier(**param)
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