import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
from lightgbm import LGBMClassifier
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data_loco
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import torch
from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score
)
# -----------------------------
# Setup
# -----------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

TARGET = VARIABLES[-1]
FEATURES = VARIABLES[:-1]

cat_cols = CAL_COLS

df = pd.read_csv("./dhs_country_file_summary.csv")

# -----------------------------
# Metric
# -----------------------------
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_all_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
    }




# -----------------------------
# Results
# -----------------------------
# rev_results = {
#     "train_country": [],
#     "test_country": [],

#     "lg_f1": [], "lgbm_f1": [], "xgb_f1": [], "tabpfn_f1": []
# }

results = {
    "train_country": [],
    "test_country": [],

    "lg_accuracy": [], "lg_f1": [], "lg_sensitivity": [], "lg_specificity": [], "lg_auc": [],
    "lg_precision": [], "lg_bal_acc": [],

    "lgbm_accuracy": [], "lgbm_f1": [], "lgbm_sensitivity": [], "lgbm_specificity": [], "lgbm_auc": [],
    "lgbm_precision": [], "lgbm_bal_acc": [],

    "xgb_accuracy": [], "xgb_f1": [], "xgb_sensitivity": [], "xgb_specificity": [], "xgb_auc": [],
    "xgb_precision": [], "xgb_bal_acc": [],

    "tabpfn_accuracy": [], "tabpfn_f1": [], "tabpfn_sensitivity": [], "tabpfn_specificity": [], "tabpfn_auc": [],
    "tabpfn_precision": [], "tabpfn_bal_acc": [],
}

with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)

# -----------------------------
# Reverse LOCO
# -----------------------------
for train_idx, train_row in df.iterrows():

    train_country = train_row["country"]
    print(f"\nTRAIN: {train_country}")

    train_dict = load_dhs_kids_data(df, indx=train_idx, variables=VARIABLES)
    train_df = preprocess_dhs(train_dict["kr"], FEATURES, CAL_COLS, TARGET)

    X_train, y_train, class_weights, train_mean, train_std = \
        prepare_ml_data_loco(train_df, FEATURES, cat_cols, TARGET)

    # -----------------------------
    # Grid models (fit ONCE per train country)
    # -----------------------------
    logreg_params = best_hyperparams["logreg"]

    lg_model = LogisticRegression(
        max_iter=2000,
        **logreg_params
    )
    lg_model.fit(X_train, y_train)

    lgbm_params = best_hyperparams["lgbm"]

    lgbm_model = LGBMClassifier(
        force_row_wise=True,
        max_bin=63,
        n_jobs=4,
        verbosity=-1,
        **lgbm_params
    )

    lgbm_model.fit(X_train, y_train)



    xgb_params = best_hyperparams["xgb"]

    base = {
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 4,
    }

    if torch.cuda.is_available():
        base["device"] = "cuda"
    else:
        base["tree_method"] = "hist"

    xgb_model = XGBClassifier(
        **base,
        **xgb_params
    )

    xgb_model.fit(X_train, y_train)

    tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
    tabpfn.fit(X_train, y_train)

    # -----------------------------
    # TEST ALL OTHER COUNTRIES
    # -----------------------------
    for test_idx in df[df["country"] != train_country].index:

        test_country = df.loc[test_idx, "country"]

        test_dict = load_dhs_kids_data(df, indx=test_idx, variables=VARIABLES)
        test_df = preprocess_dhs(test_dict["kr"], FEATURES, CAL_COLS, TARGET)

        X_test, y_test, _, _, _ = prepare_ml_data_loco(
            test_df,
            FEATURES,
            cat_cols,
            TARGET,
            train_mean=train_mean,
            train_std=train_std
        )

        results["train_country"].append(train_country)
        results["test_country"].append(test_country)

        # -----------------------------
        # Logistic Regression
        # -----------------------------
        y_pred = lg_model.predict(X_test)
        y_prob = lg_model.predict_proba(X_test)[:, 1]
        metrics = compute_all_metrics(y_test, y_pred, y_prob)

        results["lg_accuracy"].append(metrics["accuracy"])
        results["lg_f1"].append(metrics["f1"])
        results["lg_sensitivity"].append(metrics["sensitivity"])
        results["lg_specificity"].append(metrics["specificity"])
        results["lg_precision"].append(metrics["precision"])
        results["lg_auc"].append(metrics["auc"])
        results["lg_bal_acc"].append(metrics["bal_acc"])

        # -----------------------------
        # LightGBM
        # -----------------------------
        y_pred = lgbm_model.predict(X_test)
        y_prob = lgbm_model.predict_proba(X_test)[:, 1]
        metrics = compute_all_metrics(y_test, y_pred, y_prob)

        results["lgbm_accuracy"].append(metrics["accuracy"])
        results["lgbm_f1"].append(metrics["f1"])
        results["lgbm_sensitivity"].append(metrics["sensitivity"])
        results["lgbm_specificity"].append(metrics["specificity"])
        results["lgbm_precision"].append(metrics["precision"])
        results["lgbm_auc"].append(metrics["auc"])
        results["lgbm_bal_acc"].append(metrics["bal_acc"])

        # -----------------------------
        # XGBoost
        # -----------------------------
        y_pred = xgb_model.predict(X_test)
        y_prob = xgb_model.predict_proba(X_test)[:, 1]
        metrics = compute_all_metrics(y_test, y_pred, y_prob)

        results["xgb_accuracy"].append(metrics["accuracy"])
        results["xgb_f1"].append(metrics["f1"])
        results["xgb_sensitivity"].append(metrics["sensitivity"])
        results["xgb_specificity"].append(metrics["specificity"])
        results["xgb_precision"].append(metrics["precision"])
        results["xgb_auc"].append(metrics["auc"])
        results["xgb_bal_acc"].append(metrics["bal_acc"])

        # -----------------------------
        # TabPFN
        # -----------------------------
        y_pred = tabpfn.predict(X_test)
        y_prob = tabpfn.predict_proba(X_test)

        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        metrics = compute_all_metrics(y_test, y_pred, y_prob)

        results["tabpfn_accuracy"].append(metrics["accuracy"])
        results["tabpfn_f1"].append(metrics["f1"])
        results["tabpfn_sensitivity"].append(metrics["sensitivity"])
        results["tabpfn_specificity"].append(metrics["specificity"])
        results["tabpfn_precision"].append(metrics["precision"])
        results["tabpfn_auc"].append(metrics["auc"])
        results["tabpfn_bal_acc"].append(metrics["bal_acc"])

    # -----------------------------
    # SAVE ONCE (IMPORTANT FIX)
    # -----------------------------
    rev_df = pd.DataFrame(results)
    rev_df.to_csv(f"{RESULTS_DIR}/anemia_reverse_loco_results.csv", index=False)

print("DONE — Reverse LOCO saved")