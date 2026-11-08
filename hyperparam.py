import json
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
import optuna
from optuna.samplers import TPESampler

import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from config import VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Setup
# -----------------------------
TARGET = VARIABLES[-1]
FEATURES = VARIABLES[:-1]

df = pd.read_csv("./dhs_country_file_summary.csv")

# -----------------------------
# LOAD + MERGE ALL COUNTRIES
# -----------------------------
all_data = []

for idx in range(len(df)):
    data_dict = load_dhs_kids_data(df, indx=idx, variables=VARIABLES)
    all_data.append(data_dict["kr"])

df_all = pd.concat(all_data, ignore_index=True)

df_clean = preprocess_dhs(df_all, FEATURES, CAL_COLS, TARGET)

X_train, X_test, y_train, y_test, class_weights = prepare_ml_data(
    df_clean, FEATURES, CAL_COLS, TARGET
)

# -----------------------------
# CV
# -----------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# -----------------------------
# Utils
# -----------------------------
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =========================================================
# OPTUNA: Logistic Regression
# =========================================================
def tune_logreg(X, y, n_trials=30):

    def objective(trial):
        model = LogisticRegression(
            max_iter=2000,
            C=trial.suggest_float("C", 1e-3, 10.0, log=True),
            solver=trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"])
        )

        return cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    return dict(study.best_params)


# =========================================================
# OPTUNA: LightGBM
# =========================================================
def tune_lgbm(X, y, n_trials=30):

    def objective(trial):
        model = LGBMClassifier(
            force_row_wise=True,
            max_bin=63,
            n_jobs=1,
            verbosity=-1,
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 128),
            max_depth=trial.suggest_categorical("max_depth", [-1, 3, 5, 7, 10]),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )

        return cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    return dict(study.best_params)


# =========================================================
# OPTUNA: XGBoost
# =========================================================
def tune_xgb(X, y, n_trials=30):

    base = {
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 1,
    }

    if torch.cuda.is_available():
        base["device"] = "cuda"
    else:
        base["tree_method"] = "hist"

    def objective(trial):
        model = XGBClassifier(
            **base,
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )

        return cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    return dict(study.best_params)


# -----------------------------
# RUN GLOBAL OPTIMIZATION
# -----------------------------
best_hyperparams = {}

best_hyperparams["logreg"] = to_jsonable(
    tune_logreg(X_train, y_train, n_trials=30)
)

best_hyperparams["lgbm"] = to_jsonable(
    tune_lgbm(X_train, y_train, n_trials=30)
)

best_hyperparams["xgb"] = to_jsonable(
    tune_xgb(X_train, y_train, n_trials=30)
)

# -----------------------------
# SAVE ONLY HYPERPARAMETERS
# -----------------------------
hyper_path = Path(f"{RESULTS_DIR}/best_hyperparameters.json")

with open(hyper_path, "w") as f:
    json.dump(best_hyperparams, f, indent=2)

print(f"\nSaved hyperparameters to: {hyper_path}")