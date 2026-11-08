import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch

from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
print("Using device:", device)

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
TARGET   = VARIABLES[-1]
FEATURES = VARIABLES[:-1]

df = pd.read_csv("./dhs_country_file_summary.csv")
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)

# Few-shot levels — fraction of data used for training
SHOT_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]

# Confidence interval settings
CONF_LEVEL  = 0.95
CI_METHOD   = "bootstrap_bca"
N_RESAMPLES = 5000

# -----------------------------------------------------------------------------
# CI helper
# Returns a flat dict of metric values + ci_lower + ci_upper.
# Gracefully handles edge cases (single class in y_true → NaN CIs for AUC).
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    """
    Computes all five metrics + 95 % bootstrap BCA CIs.
    Returns a flat dict ready to be merged into a result row.
    """
    out = {}

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

    # AUC is undefined when only one class is present in y_true
    if len(np.unique(y_true)) < 2:
        auc, auc_lo, auc_hi = np.nan, np.nan, np.nan
    else:
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
        out[metric]               = val
        out[f"{metric}_ci_lower"] = lo
        out[f"{metric}_ci_upper"] = hi

    return out


# -----------------------------------------------------------------------------
# Main loop — countries
# -----------------------------------------------------------------------------
for index, row in df.iterrows():

    country   = row["country"]
    save_name = country.lower().replace(" ", "_")
    save_file = f"{RESULTS_DIR}/{save_name}_fewshot_results.csv"

    # ── Skip if already fully evaluated ──────────────────────────────────────
    if Path(save_file).exists():
        existing = pd.read_csv(save_file)
        if 0.90 in existing["shot"].values:
            print(f"Skipping {country} — already fully evaluated.")
            continue
        else:
            print(f"\nResuming {country} — shot=0.90 not yet reached.")
    else:
        print(f"\nProcessing: {country}")
    # ─────────────────────────────────────────────────────────────────────────

    data_dict   = load_dhs_kids_data(df, indx=index, variables=VARIABLES)
    df_clean    = preprocess_dhs(data_dict["kr"], FEATURES, CAL_COLS, TARGET)

    n_total = len(df_clean)
    print(f"  Dataset size: {n_total}")

    country_results = []

    # -------------------------------------------------------------------------
    # Inner loop — shot sizes
    # -------------------------------------------------------------------------
    for shot in SHOT_LEVELS:

        X_train, X_test, y_train, y_test, class_weights = prepare_ml_data(
            df_clean, FEATURES, CAL_COLS, TARGET,
            test_size=1 - shot,
        )
        n_shot = len(X_train)
        n_test = len(X_test)

        print(f"  shot={shot:.2f} | n_train={n_shot} | n_test={n_test}")

        # Skip degenerate splits
        if n_shot < 20 or n_test < 20:
            print("  Skipping — split too small")
            continue

        # Base row fields shared across all models for this shot level
        base_row = {
            "country": country,
            "shot":    shot,
            "n_shot":  n_shot,
            "n_test":  n_test,
        }

        # ── 1. Logistic Regression ───────────────────────────────────────────
        logreg = LogisticRegression(max_iter=1000, **best_hyperparams["logreg"])
        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)
        y_prob = logreg.predict_proba(X_test)[:, 1]

        country_results.append({
            **base_row,
            "model": "LogReg",
            **compute_metrics(y_test, y_pred, y_prob),
        })

        # ── 2. LightGBM ──────────────────────────────────────────────────────
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

        country_results.append({
            **base_row,
            "model": "LightGBM",
            **compute_metrics(y_test, y_pred, y_prob),
        })

        # ── 3. XGBoost ───────────────────────────────────────────────────────
        xgb_base = {"eval_metric": "logloss", "random_state": 42, "n_jobs": 4}
        if torch.cuda.is_available():
            xgb_base["device"] = "cuda"
        else:
            xgb_base["tree_method"] = "hist"

        xgb = XGBClassifier(**xgb_base, **best_hyperparams["xgb"])
        xgb.fit(X_train, y_train)

        y_pred = xgb.predict(X_test)
        y_prob = xgb.predict_proba(X_test)[:, 1]

        country_results.append({
            **base_row,
            "model": "XGBoost",
            **compute_metrics(y_test, y_pred, y_prob),
        })

        # ── 4. TabPFN ────────────────────────────────────────────────────────
        print("  → Training TabPFN...")
        tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
        tabpfn.fit(X_train, y_train)

        y_pred = tabpfn.predict(X_test)
        y_prob = tabpfn.predict_proba(X_test)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        country_results.append({
            **base_row,
            "model": "TabPFN",
            **compute_metrics(y_test, y_pred, y_prob),
        })

        # ── Checkpoint after every shot level ────────────────────────────────
        pd.DataFrame(country_results).to_csv(save_file, index=False)

    print(f"  Saved: {save_file}")

print("\nDONE")