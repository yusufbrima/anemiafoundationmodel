import pandas as pd
import numpy as np
from pathlib import Path
import json
from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data_loco

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

import torch

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

cat_cols = [
    "b4", "h22", "h11", "h43", "h34",
    "v190", "v113", "v116", "v106",
    "v025", "v151", "v463z", "h31"
]

output_file = "./dhs_country_file_summary.csv"
df = pd.read_csv(output_file)

# -----------------------------
# Resume previous results
# -----------------------------
results_path = f"{RESULTS_DIR}/anemia_loco_results.csv"

try:
    existing_results = pd.read_csv(results_path)
    done_countries = set(existing_results["test_country"].dropna().tolist())
    print(f"Resuming run — already completed: {len(done_countries)} countries")
except FileNotFoundError:
    existing_results = None
    done_countries = set()
    print("No existing results found — starting fresh")

# -----------------------------
# Metrics
# -----------------------------
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# -----------------------------
# Load hyperparameters
# -----------------------------
with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)

# -----------------------------
# LOCO LOOP
# -----------------------------
for test_idx, test_row in df.iterrows():

    test_country = test_row["country"]

    if test_country in done_countries:
        print(f"Skipping already processed country: {test_country}")
        continue

    print(f"\n===== LOCO TEST COUNTRY: {test_country} =====")

    # -----------------------------
    # TRAIN countries
    # -----------------------------
    train_indices = df[df["country"] != test_country].index

    train_dfs = []
    for idx in train_indices:
        data_dict = load_dhs_kids_data(df, indx=idx, variables=VARIABLES)
        kr = data_dict["kr"]

        kr_clean = preprocess_dhs(kr, FEATURES, CAL_COLS, TARGET)
        kr_clean["country"] = df.loc[idx, "country"]

        train_dfs.append(kr_clean)

    train_df = pd.concat(train_dfs, axis=0)

    # -----------------------------
    # TEST country
    # -----------------------------
    test_dict = load_dhs_kids_data(df, indx=test_idx, variables=VARIABLES)
    test_kr = test_dict["kr"]

    test_df = preprocess_dhs(test_kr, FEATURES, CAL_COLS, TARGET)

    # -----------------------------
    # PREPROCESS
    # -----------------------------
    X_train, y_train, class_weights, train_mean, train_std = \
        prepare_ml_data_loco(train_df, FEATURES, cat_cols, TARGET)

    X_test, y_test, _, _, _ = prepare_ml_data_loco(
        test_df,
        FEATURES,
        cat_cols,
        TARGET,
        train_mean=train_mean,
        train_std=train_std
    )

    # -----------------------------
    # Results for this country
    # -----------------------------
    row_result = {
        "test_country": test_country
    }

    # =========================================================
    # 1. LOGISTIC REGRESSION
    # =========================================================
    logreg = LogisticRegression(
        max_iter=2000,
        **best_hyperparams["logreg"]
    )
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]

    row_result.update({
        "lg_accuracy": accuracy_score(y_test, y_pred),
        "lg_f1": f1_score(y_test, y_pred),
        "lg_sensitivity": recall_score(y_test, y_pred),
        "lg_specificity": specificity(y_test, y_pred),
        "lg_auc": roc_auc_score(y_test, y_prob)
    })

    # =========================================================
    # 2. LIGHTGBM
    # =========================================================
    lgbm = LGBMClassifier(
        force_row_wise=True,
        max_bin=63,
        n_jobs=4,
        verbosity=-1,
        **best_hyperparams["lgbm"]
    )
    lgbm.fit(X_train, y_train)

    y_pred = lgbm.predict(X_test)
    y_prob = lgbm.predict_proba(X_test)[:, 1]

    row_result.update({
        "lgbm_accuracy": accuracy_score(y_test, y_pred),
        "lgbm_f1": f1_score(y_test, y_pred),
        "lgbm_sensitivity": recall_score(y_test, y_pred),
        "lgbm_specificity": specificity(y_test, y_pred),
        "lgbm_auc": roc_auc_score(y_test, y_prob)
    })

    # =========================================================
    # 3. XGBOOST
    # =========================================================
    base = {
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 4,
    }

    if torch.cuda.is_available():
        base["device"] = "cuda"
    else:
        base["tree_method"] = "hist"

    xgb = XGBClassifier(
        **base,
        **best_hyperparams["xgb"]
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    row_result.update({
        "xgb_accuracy": accuracy_score(y_test, y_pred),
        "xgb_f1": f1_score(y_test, y_pred),
        "xgb_sensitivity": recall_score(y_test, y_pred),
        "xgb_specificity": specificity(y_test, y_pred),
        "xgb_auc": roc_auc_score(y_test, y_prob)
    })

    # =========================================================
    # 4. TABPFN
    # =========================================================
    tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
    tabpfn.fit(X_train, y_train)

    y_pred = tabpfn.predict(X_test)
    y_prob = tabpfn.predict_proba(X_test)

    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    row_result.update({
        "tabpfn_accuracy": accuracy_score(y_test, y_pred),
        "tabpfn_f1": f1_score(y_test, y_pred),
        "tabpfn_sensitivity": recall_score(y_test, y_pred),
        "tabpfn_specificity": specificity(y_test, y_pred),
        "tabpfn_auc": roc_auc_score(y_test, y_prob)
    })

    # -----------------------------
    # Append + Save immediately
    # -----------------------------
    new_row_df = pd.DataFrame([row_result])

    if existing_results is not None:
        existing_results = pd.concat([existing_results, new_row_df], ignore_index=True)
    else:
        existing_results = new_row_df

    # remove duplicates just in case
    existing_results = existing_results.drop_duplicates(
        subset=["test_country"], keep="last"
    )

    existing_results.to_csv(results_path, index=False)

    print(f"Saved results for {test_country}")

print("\nDONE — LOCO RESULTS SAVED")
print(existing_results.head())