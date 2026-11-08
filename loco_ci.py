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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

from confidenceinterval import (
    roc_auc_score  as ci_auc_score,
    f1_score       as ci_f1_score,
    accuracy_score as ci_accuracy_score,
    tpr_score      as ci_tpr_score,   # sensitivity = TPR
    tnr_score      as ci_tnr_score,   # specificity = TNR
)

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
TARGET   = VARIABLES[-1]
FEATURES = VARIABLES[:-1]
cat_cols = CAL_COLS

df = pd.read_csv("./dhs_country_file_summary.csv")

# Confidence interval settings
CONF_LEVEL  = 0.95
CI_METHOD   = "bootstrap_bca"
N_RESAMPLES = 5000

# -----------------------------------------------------------------------------
# CI helper — mirrors compute_and_store from train_models.py
# -----------------------------------------------------------------------------
def compute_and_store(results, prefix, y_true, y_pred, y_prob):
    """
    Computes all five metrics with 95% CIs using the confidenceinterval library
    and appends point estimate + lower/upper bounds into the results dict.
    """
    acc,  (acc_lo,  acc_hi)  = ci_accuracy_score(
        y_true, y_pred,
        confidence_level=CONF_LEVEL,
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
    )
    f1,   (f1_lo,   f1_hi)   = ci_f1_score(
        y_true, y_pred,
        confidence_level=CONF_LEVEL,
        average="binary",
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
    )
    sens, (sens_lo, sens_hi) = ci_tpr_score(
        y_true, y_pred,
        confidence_level=CONF_LEVEL,
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
    )
    spec, (spec_lo, spec_hi) = ci_tnr_score(
        y_true, y_pred,
        confidence_level=CONF_LEVEL,
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
    )
    auc,  (auc_lo,  auc_hi)  = ci_auc_score(
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


# -----------------------------------------------------------------------------
# Results schema
# -----------------------------------------------------------------------------
MODELS  = ["lg", "lgbm", "xgb", "tabpfn"]
METRICS = ["accuracy", "f1", "sensitivity", "specificity", "auc"]

def init_results():
    r = {"test_country": []}
    for model in MODELS:
        for metric in METRICS:
            r[f"{model}_{metric}"]          = []
            r[f"{model}_{metric}_ci_lower"] = []
            r[f"{model}_{metric}_ci_upper"] = []
    return r

# -----------------------------------------------------------------------------
# Init or resume
# -----------------------------------------------------------------------------
results_path = Path(f"{RESULTS_DIR}/anemia_loco_results.csv")

if results_path.exists() and results_path.stat().st_size > 0:
    existing_df  = pd.read_csv(results_path)
    processed    = set(existing_df["test_country"])
    loco_results = existing_df.to_dict(orient="list")
    print(f"Resuming run — {len(processed)} countries already done")
else:
    processed    = set()
    loco_results = init_results()
    print("Starting fresh run")

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)

# -----------------------------------------------------------------------------
# Main LOCO loop
# -----------------------------------------------------------------------------
total_countries = len(df)

for i, (test_idx, test_row) in enumerate(
    tqdm(df.iterrows(), total=total_countries, desc="LOCO Countries"),
    start=1,
):
    test_country = test_row["country"]

    if test_country in processed:
        continue

    print(f"\n{'='*60}")
    print(f"[{i}/{total_countries}]  {test_country}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # Build train set (all countries except test_country)
    # -------------------------------------------------------------------------
    train_dfs = []
    for idx in df[df["country"] != test_country].index:
        data_dict = load_dhs_kids_data(df, indx=idx, variables=VARIABLES)
        kr_clean  = preprocess_dhs(data_dict["kr"], FEATURES, CAL_COLS, TARGET)
        kr_clean["country"] = df.loc[idx, "country"]
        train_dfs.append(kr_clean)

    train_df = pd.concat(train_dfs, axis=0)

    # Build test set
    test_dict = load_dhs_kids_data(df, indx=test_idx, variables=VARIABLES)
    test_df   = preprocess_dhs(test_dict["kr"], FEATURES, CAL_COLS, TARGET)

    # -------------------------------------------------------------------------
    # Preprocess — fit scaler on train, apply to test
    # -------------------------------------------------------------------------
    X_train, y_train, _, train_mean, train_std = \
        prepare_ml_data_loco(train_df, FEATURES, cat_cols, TARGET)

    X_test, y_test, _, _, _ = prepare_ml_data_loco(
        test_df, FEATURES, cat_cols, TARGET,
        train_mean=train_mean,
        train_std=train_std,
    )

    # -------------------------------------------------------------------------
    # 1. Logistic Regression
    # -------------------------------------------------------------------------
    print("   → Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=2000, **best_hyperparams["logreg"])
    logreg.fit(X_train, y_train)
    print("   ✔ Done")

    lg_pred = logreg.predict(X_test)
    lg_prob = logreg.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # 2. LightGBM
    # -------------------------------------------------------------------------
    print("   → Training LightGBM...")
    lgbm = LGBMClassifier(
        force_row_wise=True,
        max_bin=63,
        n_jobs=4,
        verbosity=-1,
        **best_hyperparams["lgbm"],
    )
    lgbm.fit(X_train, y_train)
    print("   ✔ Done")

    lgbm_pred = lgbm.predict(X_test)
    lgbm_prob = lgbm.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # 3. XGBoost  (CuPy GPU acceleration when available)
    # -------------------------------------------------------------------------
    print("   → Training XGBoost...")
    xgb_params = best_hyperparams["xgb"].copy()
    xgb_params["tree_method"] = "hist"
    if torch.cuda.is_available():
        xgb_params["device"] = "cuda"

    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        **xgb_params,
    )

    use_gpu      = torch.cuda.is_available()
    X_train_xgb  = cp.asarray(X_train) if use_gpu else X_train
    X_test_xgb   = cp.asarray(X_test)  if use_gpu else X_test

    xgb.fit(X_train_xgb, y_train)
    print("   ✔ Done")

    xgb_pred = xgb.predict(X_test_xgb)
    xgb_prob = xgb.predict_proba(X_test_xgb)
    if hasattr(xgb_prob, "get"):   # CuPy array → numpy
        xgb_prob = xgb_prob.get()
    if xgb_prob.ndim == 2:
        xgb_prob = xgb_prob[:, 1]

    # xgb_pred may also be a CuPy array; coerce to numpy for sklearn metrics
    if hasattr(xgb_pred, "get"):
        xgb_pred = xgb_pred.get()

    # -------------------------------------------------------------------------
    # 4. TabPFN
    # -------------------------------------------------------------------------
    print("   → Training TabPFN...")
    tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
    t0 = time.time()
    tabpfn.fit(X_train, y_train)

    tab_pred = tabpfn.predict(X_test)
    tab_prob = tabpfn.predict_proba(X_test)
    if tab_prob.ndim == 2:
        tab_prob = tab_prob[:, 1]

    print(f"   ✔ Done  ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------------------------
    # Store metrics + CIs for all four models
    # -------------------------------------------------------------------------
    loco_results["test_country"].append(test_country)

    compute_and_store(loco_results, "lg",     y_test, lg_pred,   lg_prob)
    compute_and_store(loco_results, "lgbm",   y_test, lgbm_pred, lgbm_prob)
    compute_and_store(loco_results, "xgb",    y_test, xgb_pred,  xgb_prob)
    compute_and_store(loco_results, "tabpfn", y_test, tab_pred,  tab_prob)

    # -------------------------------------------------------------------------
    # Checkpoint save after every country
    # -------------------------------------------------------------------------
    pd.DataFrame(loco_results).to_csv(results_path, index=False)
    processed.add(test_country)

    # Sanity print — AUC + CI for all models
    for prefix, label in [("lg", "LogReg"), ("lgbm", "LightGBM"),
                           ("xgb", "XGBoost"), ("tabpfn", "TabPFN")]:
        auc = loco_results[f"{prefix}_auc"][-1]
        lo  = loco_results[f"{prefix}_auc_ci_lower"][-1]
        hi  = loco_results[f"{prefix}_auc_ci_upper"][-1]
        print(f"  {label:10s} AUC: {auc:.3f}  95% CI [{lo:.3f} – {hi:.3f}]")

    print(f"\n✔ Saved {test_country} | Progress: {len(processed)}/{total_countries}")

print("\nDONE — ALL LOCO EXPERIMENTS COMPLETE")