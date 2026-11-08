import pandas as pd
import numpy as np
import json
import os
import torch
import shap

from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

from confidenceinterval import (
    roc_auc_score as ci_auc_score,
    f1_score      as ci_f1_score,
    accuracy_score as ci_accuracy_score,
    tpr_score     as ci_tpr_score,   # sensitivity = true positive rate
    tnr_score     as ci_tnr_score,   # specificity = true negative rate
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
os.environ["TABPFN_API_KEY"] = "tabpfn_sk_8VH-7Gyr5PHkSIeJsJMRYTlwyQn9vHMmJG5ZitiUi5M"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TARGET   = VARIABLES[-1]
FEATURES = VARIABLES[:-1]

# Confidence interval settings
CONF_LEVEL  = 0.95
CI_METHOD   = "bootstrap_bca"   # used for accuracy, f1, sensitivity, specificity
N_RESAMPLES = 5000

# -----------------------------------------------------------------------------
# Load data and hyperparameters
# -----------------------------------------------------------------------------
df = pd.read_csv("./dhs_country_file_summary.csv")

with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)

# -----------------------------------------------------------------------------
# Results container — wide format with point estimate + CI bounds per metric
# -----------------------------------------------------------------------------
MODELS  = ["lg", "lgbm", "xgb", "tabpfn"]
METRICS = ["accuracy", "f1", "sensitivity", "specificity", "auc"]

results = {"country": []}
for model in MODELS:
    for metric in METRICS:
        results[f"{model}_{metric}"]          = []
        results[f"{model}_{metric}_ci_lower"] = []
        results[f"{model}_{metric}_ci_upper"] = []

shap_results = {"model": [], "country": [], "feature": [], "importance": []}


# -----------------------------------------------------------------------------
# Helper: compute all metrics + CIs and store in results dict
# -----------------------------------------------------------------------------
def compute_and_store(results, prefix, y_true, y_pred, y_prob):
    """
    Computes all five metrics with 95% CIs using the confidenceinterval library
    and appends point estimate + lower/upper bounds into the results dict.

    - accuracy, sensitivity, specificity: bootstrap_bca (binomial proportions)
    - f1:  bootstrap_bca
    - auc: bootstrap_bca on y_prob (DeLong analytic also available by default)
    """

    # accuracy
    acc, (acc_lo, acc_hi) = ci_accuracy_score(
        y_true, y_pred,
        confidence_level=CONF_LEVEL,
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
    )

    # F1 (binary)
    f1, (f1_lo, f1_hi) = ci_f1_score(
        y_true, y_pred,
        confidence_level=CONF_LEVEL,
        average="binary",
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
    )

    # sensitivity = TPR
    sens, (sens_lo, sens_hi) = ci_tpr_score(
        y_true, y_pred,
        confidence_level=CONF_LEVEL,
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
    )

    # specificity = TNR
    spec, (spec_lo, spec_hi) = ci_tnr_score(
        y_true, y_pred,
        confidence_level=CONF_LEVEL,
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
    )

    # AUC-ROC — pass y_prob (continuous scores), not hard labels
    auc, (auc_lo, auc_hi) = ci_auc_score(
        y_true, y_prob,
        confidence_level=CONF_LEVEL,
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
    )

    for metric, val, lo, hi in [
        ("accuracy",    acc,  acc_lo,  acc_hi),
        ("f1",          f1,   f1_lo,   f1_hi),
        ("sensitivity", sens, sens_lo, sens_hi),
        ("specificity", spec, spec_lo, spec_hi),
        ("auc",         auc,  auc_lo,  auc_hi),
    ]:
        results[f"{prefix}_{metric}"].append(val)
        results[f"{prefix}_{metric}_ci_lower"].append(lo)
        results[f"{prefix}_{metric}_ci_upper"].append(hi)


def store_shap(feature_names, shap_mean_imp, model_name, country):
    for f, v in zip(feature_names, shap_mean_imp):
        shap_results["model"].append(model_name)
        shap_results["feature"].append(f)
        shap_results["country"].append(country)
        shap_results["importance"].append(v)


# -----------------------------------------------------------------------------
# Main loop — iterate over countries
# -----------------------------------------------------------------------------
for index, row in df.iterrows():

    country = row["country"]
    print(f"\n{'='*60}")
    print(f"Processing: {country}")
    print(f"{'='*60}")

    data_dict = load_dhs_kids_data(df, indx=index, variables=VARIABLES)
    df_clean  = preprocess_dhs(data_dict["kr"], FEATURES, CAL_COLS, TARGET)

    X_train, X_test, y_train, y_test, class_weights = prepare_ml_data(
        df_clean, FEATURES, CAL_COLS, TARGET
    )

    results["country"].append(country)

    # -------------------------------------------------------------------------
    # 1. Logistic Regression
    # -------------------------------------------------------------------------
    print("Training Logistic Regression...")

    logreg = LogisticRegression(max_iter=2000, **best_hyperparams["logreg"])
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]

    compute_and_store(results, "lg", y_test, y_pred, y_prob)

    shap_vals = shap.LinearExplainer(logreg, X_test).shap_values(X_test)
    store_shap(FEATURES, np.abs(shap_vals).mean(axis=0), "LogReg", country)

    # -------------------------------------------------------------------------
    # 2. LightGBM
    # -------------------------------------------------------------------------
    print("Training LightGBM...")

    lgbm = LGBMClassifier(
        force_row_wise=True,
        max_bin=63,
        n_jobs=4,
        verbosity=-1,
        **best_hyperparams["lgbm"],
    )
    lgbm.fit(X_train, y_train)

    y_pred = lgbm.predict(X_test)
    y_prob = lgbm.predict_proba(X_test)[:, 1]

    compute_and_store(results, "lgbm", y_test, y_pred, y_prob)

    shap_vals = shap.TreeExplainer(lgbm).shap_values(X_test)
    store_shap(FEATURES, np.abs(shap_vals).mean(axis=0), "LightGBM", country)

    # -------------------------------------------------------------------------
    # 3. XGBoost
    # -------------------------------------------------------------------------
    print("Training XGBoost...")

    xgb_base = {
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 4,
    }
    if torch.cuda.is_available():
        xgb_base["device"] = "cuda"
    else:
        xgb_base["tree_method"] = "hist"

    xgb = XGBClassifier(**xgb_base, **best_hyperparams["xgb"])
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    compute_and_store(results, "xgb", y_test, y_pred, y_prob)

    shap_vals = shap.TreeExplainer(xgb).shap_values(X_test)
    store_shap(FEATURES, np.abs(shap_vals).mean(axis=0), "XGBoost", country)

    # -------------------------------------------------------------------------
    # 4. TabPFN
    # -------------------------------------------------------------------------
    print("Training TabPFN...")

    tabpfn = TabPFNClassifier(ignore_pretraining_limits=True, device="cuda")
    tabpfn.to(device)
    tabpfn.fit(X_train, y_train)

    y_pred = tabpfn.predict(X_test)
    y_prob = tabpfn.predict_proba(X_test)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    compute_and_store(results, "tabpfn", y_test, y_pred, y_prob)

    r = permutation_importance(
        tabpfn, X_test, y_test,
        n_repeats=5,
        random_state=42,
        scoring="f1",
    )
    store_shap(FEATURES, r.importances_mean, "TabPFN", country)

    # -------------------------------------------------------------------------
    # Save after every country — safe to resume if interrupted
    # -------------------------------------------------------------------------
    pd.DataFrame(results).to_csv(
        f"{RESULTS_DIR}/anemia_model_results_by_country.csv", index=False
    )
    pd.DataFrame(shap_results).to_csv(
        f"{RESULTS_DIR}/anemia_model_results_by_country_shap.csv", index=False
    )

    # Sanity print — AUC with CI for each model
    for prefix, label in [("lg", "LogReg"), ("lgbm", "LightGBM"),
                           ("xgb", "XGBoost"), ("tabpfn", "TabPFN")]:
        auc = results[f"{prefix}_auc"][-1]
        lo  = results[f"{prefix}_auc_ci_lower"][-1]
        hi  = results[f"{prefix}_auc_ci_upper"][-1]
        print(f"  {label:10s} AUC: {auc:.3f}  95% CI [{lo:.3f} – {hi:.3f}]")

print("\nDone. Results saved to:", RESULTS_DIR)