import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data
import json
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from scipy.special import logit

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

output_file = "./dhs_country_file_summary.csv"
df = pd.read_csv(output_file)

# -----------------------------
# Results container (WIDE FORMAT)
# -----------------------------
results = {
    "country": [],

    "lg_brier": [], "lg_calib_intercept": [], "lg_calib_slope": [], "lg_ece": [],
    "lgbm_brier": [], "lgbm_calib_intercept": [], "lgbm_calib_slope": [], "lgbm_ece": [],
    "xgb_brier": [], "xgb_calib_intercept": [], "xgb_calib_slope": [], "xgb_ece": [],
    "tabpfn_brier": [], "tabpfn_calib_intercept": [], "tabpfn_calib_slope": [], "tabpfn_ece": [],
}

# Long-format predictions for reliability diagrams
prediction_records = []

# Optional: pre-binned calibration data for plotting
calibration_bin_records = []

# -----------------------------
# Metric helpers
# -----------------------------
def safe_probabilities(y_prob, eps=1e-15):
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    return np.clip(y_prob, eps, 1 - eps)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    ECE using equal-width bins over [0, 1].
    Lower is better. 0 is ideal.
    """
    y_true = np.asarray(y_true)
    y_prob = safe_probabilities(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return ece

def calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Returns:
      - Brier score
      - Calibration intercept
      - Calibration slope
      - ECE
    """
    y_prob = safe_probabilities(y_prob)

    brier = brier_score_loss(y_true, y_prob)

    # Calibration intercept and slope:
    # fit y_true ~ logit(pred_prob)
    lp = logit(y_prob).reshape(-1, 1)

    calib_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000
    )
    calib_model.fit(lp, y_true)

    calib_intercept = float(calib_model.intercept_[0])
    calib_slope = float(calib_model.coef_[0][0])

    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    return brier, calib_intercept, calib_slope, ece

def store_predictions(country, model_name, y_true, y_prob):
    """
    Save individual-level predictions for later calibration/reliability plots.
    """
    y_prob = safe_probabilities(y_prob)
    y_true = np.asarray(y_true)

    for yt, yp in zip(y_true, y_prob):
        prediction_records.append({
            "country": country,
            "model": model_name,
            "y_true": int(yt),
            "y_prob": float(yp)
        })

    # Also store binned calibration values for direct plotting
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile"
    )

    # Reconstruct bin membership for counts using quantile bins
    quantiles = np.quantile(y_prob, np.linspace(0, 1, 11))
    quantiles = np.unique(quantiles)

    if len(quantiles) > 2:
        bin_ids = np.digitize(y_prob, quantiles[1:-1], right=True)

        for b in range(len(quantiles) - 1):
            mask = bin_ids == b
            if np.any(mask):
                calibration_bin_records.append({
                    "country": country,
                    "model": model_name,
                    "bin": b + 1,
                    "mean_predicted_probability": float(np.mean(y_prob[mask])),
                    "observed_probability": float(np.mean(y_true[mask])),
                    "n": int(np.sum(mask))
                })


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

shap_results = {
    "model": [],
    "country": [],
    "feature": [],
    "importance": []
}


with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)
# -----------------------------
# Loop over countries
# -----------------------------
for index, row in df.iterrows():

    country = row["country"]
    print(f"\nProcessing: {country}")

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

    y_prob = logreg.predict_proba(X_test)[:, 1]
    lg_brier, lg_intercept, lg_slope, lg_ece = calibration_metrics(y_test, y_prob)

    results["lg_brier"].append(lg_brier)
    results["lg_calib_intercept"].append(lg_intercept)
    results["lg_calib_slope"].append(lg_slope)
    results["lg_ece"].append(lg_ece)

    store_predictions(country, "LogReg", y_test, y_prob)

    explainer = shap.LinearExplainer(logreg, X_test)
    shap_values = explainer.shap_values(X_test)
    mean_imp = np.abs(shap_values).mean(axis=0)

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

    y_prob = lgbm.predict_proba(X_test)[:, 1]
    lgbm_brier, lgbm_intercept, lgbm_slope, lgbm_ece = calibration_metrics(y_test, y_prob)

    results["lgbm_brier"].append(lgbm_brier)
    results["lgbm_calib_intercept"].append(lgbm_intercept)
    results["lgbm_calib_slope"].append(lgbm_slope)
    results["lgbm_ece"].append(lgbm_ece)

    store_predictions(country, "LightGBM", y_test, y_prob)

    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(X_test)
    mean_imp = np.abs(shap_values).mean(axis=0)

    # =========================================================
    # 3. XGBOOST
    # =========================================================

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

    xgb = XGBClassifier(
        **base,
        **xgb_params
    )

    xgb.fit(X_train, y_train)

    y_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_brier, xgb_intercept, xgb_slope, xgb_ece = calibration_metrics(y_test, y_prob)

    results["xgb_brier"].append(xgb_brier)
    results["xgb_calib_intercept"].append(xgb_intercept)
    results["xgb_calib_slope"].append(xgb_slope)
    results["xgb_ece"].append(xgb_ece)

    store_predictions(country, "XGBoost", y_test, y_prob)

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    mean_imp = np.abs(shap_values).mean(axis=0)

    # =========================================================
    # 4. TABPFN
    # =========================================================
    print("Training TabPFN...")

    tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
    tabpfn.fit(X_train, y_train)

    y_prob = tabpfn.predict_proba(X_test)
    y_prob = safe_probabilities(y_prob)

    tabpfn_brier, tabpfn_intercept, tabpfn_slope, tabpfn_ece = calibration_metrics(y_test, y_prob)

    results["tabpfn_brier"].append(tabpfn_brier)
    results["tabpfn_calib_intercept"].append(tabpfn_intercept)
    results["tabpfn_calib_slope"].append(tabpfn_slope)
    results["tabpfn_ece"].append(tabpfn_ece)

    store_predictions(country, "TabPFN", y_test, y_prob)

    # -----------------------------
    # Save results after each country
    # -----------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/anemia_model_calibration_by_country.csv", index=False)

    predictions_df = pd.DataFrame(prediction_records)
    predictions_df.to_csv(f"{RESULTS_DIR}/anemia_model_predictions_by_country.csv", index=False)

    calibration_bins_df = pd.DataFrame(calibration_bin_records)
    calibration_bins_df.to_csv(f"{RESULTS_DIR}/anemia_model_calibration_bins_by_country.csv", index=False)

    shap_df = pd.DataFrame(shap_results)
    # shap_df.to_csv(f"{RESULTS_DIR}/anemia_model_results_by_country_shap.csv", index=False)

    print("\nDONE. Saved:")
    print(f"- {RESULTS_DIR}/anemia_model_calibration_by_country.csv")
    print(f"- {RESULTS_DIR}/anemia_model_predictions_by_country.csv")
    print(f"- {RESULTS_DIR}/anemia_model_calibration_bins_by_country.csv")