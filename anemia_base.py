import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb_lib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import os
import torch

# set_access_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMWU2NjZjZTAtYmJkOC00Y2E5LWIxNzgtNzllYjY4YjM3MGQwIiwiZXhwIjoxODA3NTI4NTg0fQ.d7lesP0Rseri9EV4T1Ue9PDEK9Y9BOum4NRC0MQspc4")
os.environ["TABPFN_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMWU2NjZjZTAtYmJkOC00Y2E5LWIxNzgtNzllYjY4YjM3MGQwIiwiZXhwIjoxODA3NTI4NTg0fQ.d7lesP0Rseri9EV4T1Ue9PDEK9Y9BOum4NRC0MQspc4"


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

output_file = "./dhs_country_file_summary.csv"
df = pd.read_csv(output_file)

with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)


# -----------------------------
# Results container (WIDE FORMAT)
# -----------------------------
results = {
    "country": [],

    "lg_accuracy": [], "lg_f1": [], "lg_sensitivity": [], "lg_specificity": [], "lg_auc": [],
    "lgbm_accuracy": [], "lgbm_f1": [], "lgbm_sensitivity": [], "lgbm_specificity": [], "lgbm_auc": [],
    "xgb_accuracy": [], "xgb_f1": [], "xgb_sensitivity": [], "xgb_specificity": [], "xgb_auc": [],
    "tabpfn_accuracy": [], "tabpfn_f1": [], "tabpfn_sensitivity": [], "tabpfn_specificity": [], "tabpfn_auc": [],
}

# -----------------------------
# Metric helpers
# -----------------------------
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

shap_results = {
    "model": [],
    "country": [],
    "feature": [],
    "importance": []
}

# -----------------------------
# Loop over countries
# -----------------------------
for index, row in df.iterrows():

    country = row["country"]
    print(f"\nProcessing: {country}")
    # shap_results['country'].append(country)

    data_dict = load_dhs_kids_data(df, indx=index, variables=VARIABLES)
    ik_stata_df = data_dict["kr"]

    df_clean = preprocess_dhs(ik_stata_df, FEATURES, CAL_COLS, TARGET)

    X_train, X_test, y_train, y_test, class_weights = prepare_ml_data(
        df_clean, FEATURES, CAL_COLS, TARGET
    )

    results["country"].append(country)

    # =========================================================
    # 1. LOGISTIC REGRESSION
    # =========================================================
    logreg_params = best_hyperparams["logreg"]

    logreg = LogisticRegression(
        max_iter=2000,
        **logreg_params
    )

    logreg.fit(X_train, y_train)


    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]

    results["lg_accuracy"].append(accuracy_score(y_test, y_pred))
    results["lg_f1"].append(f1_score(y_test, y_pred))
    results["lg_sensitivity"].append(recall_score(y_test, y_pred))
    results["lg_specificity"].append(specificity(y_test, y_pred))
    results["lg_auc"].append(roc_auc_score(y_test, y_prob))


    explainer = shap.LinearExplainer(logreg, X_test)
    shap_values = explainer.shap_values(X_test)

    mean_imp = np.abs(shap_values).mean(axis=0)

    for f, v in zip(FEATURES, mean_imp):
        shap_results["model"].append("LogReg")
        shap_results["feature"].append(f)
        shap_results["country"].append(country)
        shap_results["importance"].append(v)
    


    # =========================================================
    # 2. LIGHTGBM
    # =========================================================

    lgbm_params = best_hyperparams["lgbm"]

    lgbm = LGBMClassifier(
        force_row_wise=True,
        max_bin=63,
        n_jobs=4,
        verbosity=-1,
        **lgbm_params
    )

    lgbm.fit(X_train, y_train)

    y_pred = lgbm.predict(X_test)
    y_prob = lgbm.predict_proba(X_test)[:, 1]


     

    results["lgbm_accuracy"].append(accuracy_score(y_test, y_pred))
    results["lgbm_f1"].append(f1_score(y_test, y_pred))
    results["lgbm_sensitivity"].append(recall_score(y_test, y_pred))
    results["lgbm_specificity"].append(specificity(y_test, y_pred))
    results["lgbm_auc"].append(roc_auc_score(y_test, y_prob))



    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_test)

    mean_imp = np.abs(shap_values).mean(axis=0)

    for f, v in zip(FEATURES, mean_imp):
        shap_results["model"].append("LightGBM")
        shap_results["feature"].append(f)
        shap_results["importance"].append(v)
        shap_results["country"].append(country)


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
        **xgb_params
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]




    results["xgb_accuracy"].append(accuracy_score(y_test, y_pred))
    results["xgb_f1"].append(f1_score(y_test, y_pred))
    results["xgb_sensitivity"].append(recall_score(y_test, y_pred))
    results["xgb_specificity"].append(specificity(y_test, y_pred))
    results["xgb_auc"].append(roc_auc_score(y_test, y_prob))

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    mean_imp = np.abs(shap_values).mean(axis=0)

    for f, v in zip(FEATURES, mean_imp):
        shap_results["model"].append("XGBoost")
        shap_results["feature"].append(f)
        shap_results["country"].append(country)
        shap_results["importance"].append(v)

    # =========================================================
    # 4. TABPFN
    # =========================================================
    print("Training TabPFN...")

    tabpfn = TabPFNClassifier(ignore_pretraining_limits=True,device="cuda",)
    tabpfn.to(device)
    tabpfn.fit(X_train, y_train)

    y_pred = tabpfn.predict(X_test)

    results["tabpfn_accuracy"].append(accuracy_score(y_test, y_pred))
    results["tabpfn_f1"].append(f1_score(y_test, y_pred))
    results["tabpfn_sensitivity"].append(recall_score(y_test, y_pred))
    results["tabpfn_specificity"].append(specificity(y_test, y_pred))

    y_prob = tabpfn.predict_proba(X_test)

    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    results["tabpfn_auc"].append(auc)


    r = permutation_importance(
        tabpfn,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="f1"
    )

    for f, v in zip(FEATURES, r.importances_mean):
        shap_results["model"].append("TabPFN")
        shap_results["feature"].append(f)
        shap_results["country"].append(country)
        shap_results["importance"].append(v)

    # -----------------------------
    # Save results
    # -----------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/anemia_model_results_by_country.csv", index=False)



    shap_df = pd.DataFrame(shap_results)

    shap_df.to_csv(f"{RESULTS_DIR}/anemia_model_results_by_country_shap.csv", index=False)

    print("\nDONE. Saved to anemia_model_results_by_country.csv")
    print(results_df.head())