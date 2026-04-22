import json
import pandas as pd
import numpy as np
import torch

from config import RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix
)

# =========================================================
# LOAD OPTUNA HYPERPARAMETERS
# =========================================================
with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    BEST_PARAMS = json.load(f)

print("Loaded hyperparameters:", BEST_PARAMS.keys())

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# SETUP
# =========================================================
TARGET = VARIABLES[-1]
FEATURES = VARIABLES[:-1]

df = pd.read_csv("./dhs_country_file_summary.csv")

# =========================================================
# SUBGROUPS
# =========================================================
SUBGROUPS = {
    "sex": "b4",
    "wealth": "v190",
    "education": "v106",
    "residence": "v025",
    "age_group": "age_group"
}

# =========================================================
# CV
# =========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =========================================================
# METRICS
# =========================================================
def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return np.nan
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)

# =========================================================
# PREPROCESSOR
# =========================================================
def build_preprocessor(X):

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ])

# =========================================================
# MODEL BUILDERS (USING OPTUNA PARAMS)
# =========================================================
def get_logreg():
    return LogisticRegression(
        max_iter=2000,
        **BEST_PARAMS["logreg"]
    )

def get_lgbm():
    return LGBMClassifier(
        force_row_wise=True,
        max_bin=63,
        n_jobs=4,
        verbosity=-1,
        **BEST_PARAMS["lgbm"]
    )

def get_xgb():
    base = {
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 4,
    }

    if torch.cuda.is_available():
        base["device"] = "cuda"
    else:
        base["tree_method"] = "hist"

    return XGBClassifier(
        **base,
        **BEST_PARAMS["xgb"]
    )

# =========================================================
# METRICS HELPERS
# =========================================================
results_rows = []

# =========================================================
# MAIN LOOP
# =========================================================
for index, row in df.iterrows():

    country = row["country"]
    print(f"\nProcessing: {country}")

    data_dict = load_dhs_kids_data(df, indx=index, variables=VARIABLES)
    ik_df = data_dict["kr"]

    df_clean = preprocess_dhs(ik_df, FEATURES, CAL_COLS, TARGET)

    if "hw1" in df_clean.columns:
        df_clean["age_group"] = pd.cut(
            df_clean["hw1"],
            bins=[0, 6, 12, 24, 36, 48, 60],
            labels=["0–5m", "6–11m", "12–23m", "24–35m", "36–47m", "48–59m"]
        )

    for col in ["b4", "v190", "v106", "v025"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)

    # =====================================================
    # SUBGROUP LOOP
    # =====================================================
    for subgroup_name, subgroup_col in SUBGROUPS.items():

        if subgroup_col not in df_clean.columns:
            continue

        values = df_clean[subgroup_col].dropna().unique()

        for val in values:

            sub_df = df_clean[df_clean[subgroup_col] == val].copy()
            sub_df = sub_df.replace([np.inf, -np.inf], np.nan)
            sub_df = sub_df.dropna(subset=[TARGET] + FEATURES + CAL_COLS)

            if len(sub_df) < 50:
                continue

            X_train, X_test, y_train, y_test, _ = prepare_ml_data(
                sub_df, FEATURES, CAL_COLS, TARGET
            )

            if len(np.unique(y_train)) < 2:
                continue

            pre = build_preprocessor(X_train)
            X_train_p = pre.fit_transform(X_train)
            X_test_p = pre.transform(X_test)

            # =================================================
            # LOGISTIC REGRESSION
            # =================================================
            lg = get_logreg()
            lg.fit(X_train_p, y_train)

            pred = lg.predict(X_test_p)
            prob = lg.predict_proba(X_test_p)[:, 1]

            results_rows.append({
                "country": country,
                "subgroup": subgroup_name,
                "value": str(val),
                "model": "LogReg",
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "sensitivity": recall_score(y_test, pred),
                "specificity": specificity(y_test, pred),
                "auc": safe_auc(y_test, prob),
            })

            # =================================================
            # LIGHTGBM
            # =================================================
            lgbm = get_lgbm()
            lgbm.fit(X_train_p, y_train)

            pred = lgbm.predict(X_test_p)
            prob = lgbm.predict_proba(X_test_p)[:, 1]

            results_rows.append({
                "country": country,
                "subgroup": subgroup_name,
                "value": str(val),
                "model": "LightGBM",
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "sensitivity": recall_score(y_test, pred),
                "specificity": specificity(y_test, pred),
                "auc": safe_auc(y_test, prob),
            })

            # =================================================
            # XGBOOST
            # =================================================
            xgb = get_xgb()
            xgb.fit(X_train_p, y_train)

            pred = xgb.predict(X_test_p)
            prob = xgb.predict_proba(X_test_p)[:, 1]

            results_rows.append({
                "country": country,
                "subgroup": subgroup_name,
                "value": str(val),
                "model": "XGBoost",
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "sensitivity": recall_score(y_test, pred),
                "specificity": specificity(y_test, pred),
                "auc": safe_auc(y_test, prob),
            })

            # =================================================
            # TABPFN (UNCHANGED)
            # =================================================
            tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
            tabpfn.fit(X_train_p, y_train)

            pred = tabpfn.predict(X_test_p)
            prob = tabpfn.predict_proba(X_test_p)[:, 1]

            results_rows.append({
                "country": country,
                "subgroup": subgroup_name,
                "value": str(val),
                "model": "TabPFN",
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "sensitivity": recall_score(y_test, pred),
                "specificity": specificity(y_test, pred),
                "auc": safe_auc(y_test, prob),
            })

# =========================================================
# SAVE FINAL RESULTS
# =========================================================
results_df = pd.DataFrame(results_rows)
results_df.to_csv(f"{RESULTS_DIR}/subgroup_analysis_optuna_models.csv", index=False)

print("\nDONE")
print(results_df.head())