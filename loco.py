import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
from tqdm import tqdm
import time

import cupy as cp


from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data_loco

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
# from tabpfn_client import set_access_token, TabPFNClassifier

# set_access_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMWU2NjZjZTAtYmJkOC00Y2E5LWIxNzgtNzllYjY4YjM3MGQwIiwiZXhwIjoxODA3NTI4NTg0fQ.d7lesP0Rseri9EV4T1Ue9PDEK9Y9BOum4NRC0MQspc4")

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
cat_cols = CAL_COLS

df = pd.read_csv("./dhs_country_file_summary.csv")

# -----------------------------
# Metrics
# -----------------------------
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# -----------------------------
# Results path
# -----------------------------
results_path = Path(f"{RESULTS_DIR}/anemia_loco_results.csv")

# -----------------------------
# Init or resume
# -----------------------------
def init_results():
    return {
        "test_country": [],
        "lg_accuracy": [], "lg_f1": [], "lg_sensitivity": [], "lg_specificity": [], "lg_auc": [],
        "lgbm_accuracy": [], "lgbm_f1": [], "lgbm_sensitivity": [], "lgbm_specificity": [], "lgbm_auc": [],
        "xgb_accuracy": [], "xgb_f1": [], "xgb_sensitivity": [], "xgb_specificity": [], "xgb_auc": [],
        "tabpfn_accuracy": [], "tabpfn_f1": [], "tabpfn_sensitivity": [], "tabpfn_specificity": [], "tabpfn_auc": [],
    }

if results_path.exists() and results_path.stat().st_size > 0:
    existing_df = pd.read_csv(results_path)
    processed = set(existing_df["test_country"])
    loco_results = existing_df.to_dict(orient="list")
    print(f"Resuming run — {len(processed)} countries already done")
else:
    processed = set()
    loco_results = init_results()
    print("Starting fresh run")

# -----------------------------
# Hyperparameters
# -----------------------------
with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)

# -----------------------------
# LOOP
# -----------------------------
total_countries = len(df)

for i, (test_idx, test_row) in enumerate(
    tqdm(df.iterrows(), total=total_countries, desc="LOCO Countries"),
    start=1
):

    test_country = test_row["country"]

    if test_country in processed:
        continue

    print(f"\n===== [{i}/{total_countries}] {test_country} =====")

    # =========================================================
    # TRAIN / TEST SPLIT
    # =========================================================
    train_indices = df[df["country"] != test_country].index

    train_dfs = []
    for idx in train_indices:
        data_dict = load_dhs_kids_data(df, indx=idx, variables=VARIABLES)
        kr = data_dict["kr"]

        kr_clean = preprocess_dhs(kr, FEATURES, CAL_COLS, TARGET)
        kr_clean["country"] = df.loc[idx, "country"]

        train_dfs.append(kr_clean)

    train_df = pd.concat(train_dfs, axis=0)

    test_dict = load_dhs_kids_data(df, indx=test_idx, variables=VARIABLES)
    test_df = preprocess_dhs(test_dict["kr"], FEATURES, CAL_COLS, TARGET)

    # =========================================================
    # PREPROCESS
    # =========================================================
    X_train, y_train, _, train_mean, train_std = \
        prepare_ml_data_loco(train_df, FEATURES, cat_cols, TARGET)

    X_test, y_test, _, _, _ = prepare_ml_data_loco(
        test_df,
        FEATURES,
        cat_cols,
        TARGET,
        train_mean=train_mean,
        train_std=train_std
    )

    # =========================================================
    # 1. LOGISTIC REGRESSION
    # =========================================================
    print("   → Training Logistic Regression...")

    logreg = LogisticRegression(max_iter=2000, **best_hyperparams["logreg"])
    logreg.fit(X_train, y_train)

    print("   ✔ Logistic Regression done")

    lg_pred = logreg.predict(X_test)
    lg_prob = logreg.predict_proba(X_test)[:, 1]

    # =========================================================
    # 2. LIGHTGBM
    # =========================================================
    print("   → Training LightGBM...")

    lgbm = LGBMClassifier(
        force_row_wise=True,
        max_bin=63,
        n_jobs=4,
        verbosity=-1,
        **best_hyperparams["lgbm"]
    )

    lgbm.fit(X_train, y_train)

    print("   ✔ LightGBM done")

    lgbm_pred = lgbm.predict(X_test)
    lgbm_prob = lgbm.predict_proba(X_test)[:, 1]

    # =========================================================
    # 3. XGBOOST (GPU SAFE + CuPy OPTIONAL)
    # =========================================================
    print("   → Training XGBoost...")

    xgb_params = best_hyperparams["xgb"].copy()

    if torch.cuda.is_available():
        xgb_params["device"] = "cuda"
        xgb_params["tree_method"] = "hist"   # XGBoost 2.x correct setting
    else:
        xgb_params["tree_method"] = "hist"

    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        **xgb_params
    )

    # CuPy (optional GPU acceleration)
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        X_train_xgb = cp.asarray(X_train)
        X_test_xgb = cp.asarray(X_test)
    else:
        X_train_xgb = X_train
        X_test_xgb = X_test

    xgb.fit(X_train_xgb, y_train)

    print("   ✔ XGBoost done")

    xgb_pred = xgb.predict(X_test_xgb)
    xgb_prob = xgb.predict_proba(X_test_xgb)

    if hasattr(xgb_prob, "get"):
        xgb_prob = xgb_prob.get()

    if xgb_prob.ndim == 2:
        xgb_prob = xgb_prob[:, 1]

    # =========================================================
    # 4. TABPFN
    # =========================================================
    print("   → Training TabPFN...")

    tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
    t0 = time.time()

    tabpfn.fit(X_train, y_train)

    # print("   ✔ TabPFN done")

    tab_pred = tabpfn.predict(X_test)

    tab_prob = tabpfn.predict_proba(X_test)
    print(f"TabPFN time: {time.time() - t0:.2f}s")
    if tab_prob.ndim == 2:
        tab_prob = tab_prob[:, 1]

    # =========================================================
    # STORE RESULTS
    # =========================================================
    loco_results["test_country"].append(test_country)

    loco_results["lg_accuracy"].append(accuracy_score(y_test, lg_pred))
    loco_results["lg_f1"].append(f1_score(y_test, lg_pred))
    loco_results["lg_sensitivity"].append(recall_score(y_test, lg_pred))
    loco_results["lg_specificity"].append(specificity(y_test, lg_pred))
    loco_results["lg_auc"].append(roc_auc_score(y_test, lg_prob))

    loco_results["lgbm_accuracy"].append(accuracy_score(y_test, lgbm_pred))
    loco_results["lgbm_f1"].append(f1_score(y_test, lgbm_pred))
    loco_results["lgbm_sensitivity"].append(recall_score(y_test, lgbm_pred))
    loco_results["lgbm_specificity"].append(specificity(y_test, lgbm_pred))
    loco_results["lgbm_auc"].append(roc_auc_score(y_test, lgbm_prob))

    loco_results["xgb_accuracy"].append(accuracy_score(y_test, xgb_pred))
    loco_results["xgb_f1"].append(f1_score(y_test, xgb_pred))
    loco_results["xgb_sensitivity"].append(recall_score(y_test, xgb_pred))
    loco_results["xgb_specificity"].append(specificity(y_test, xgb_pred))
    loco_results["xgb_auc"].append(roc_auc_score(y_test, xgb_prob))

    loco_results["tabpfn_accuracy"].append(accuracy_score(y_test, tab_pred))
    loco_results["tabpfn_f1"].append(f1_score(y_test, tab_pred))
    loco_results["tabpfn_sensitivity"].append(recall_score(y_test, tab_pred))
    loco_results["tabpfn_specificity"].append(specificity(y_test, tab_pred))
    loco_results["tabpfn_auc"].append(roc_auc_score(y_test, tab_prob))

    # =========================================================
    # SAVE CHECKPOINT
    # =========================================================
    pd.DataFrame(loco_results).to_csv(results_path, index=False)

    processed.add(test_country)

    print(f"✔ Saved {test_country} | Progress: {len(processed)}/{total_countries}")

print("\nDONE — ALL LOCO EXPERIMENTS COMPLETE")