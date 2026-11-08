import pandas as pd
import numpy as np
import json
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data_loco

from confidenceinterval import (
    roc_auc_score  as ci_auc_score,
    f1_score       as ci_f1_score,
    accuracy_score as ci_accuracy_score,
    tpr_score      as ci_tpr_score,   # sensitivity = TPR
    tnr_score      as ci_tnr_score,   # specificity = TNR
    ppv_score      as ci_precision_score,  # precision = PPV
)
from confidenceinterval.bootstrap import bootstrap_ci

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

TARGET   = VARIABLES[-1]
FEATURES = VARIABLES[:-1]
cat_cols = CAL_COLS

df = pd.read_csv("./dhs_country_file_summary.csv")

# Confidence interval settings
CONF_LEVEL  = 0.95
CI_METHOD   = "bootstrap_bca"
N_RESAMPLES = 5000

# -----------------------------------------------------------------------------
# Results schema
# NOTE: this script has two extra metrics vs the others: precision + bal_acc
# -----------------------------------------------------------------------------
MODELS  = ["lg", "lgbm", "xgb", "tabpfn"]
METRICS = ["accuracy", "f1", "sensitivity", "specificity", "precision", "auc", "bal_acc"]

results = {"train_country": [], "test_country": []}
for model in MODELS:
    for metric in METRICS:
        results[f"{model}_{metric}"]          = []
        results[f"{model}_{metric}_ci_lower"] = []
        results[f"{model}_{metric}_ci_upper"] = []

with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)

# -----------------------------------------------------------------------------
# CI helper
# Handles all seven metrics. balanced_accuracy_score is not in the
# confidenceinterval library directly, so we use its generic bootstrap_ci
# with the sklearn callable.
# -----------------------------------------------------------------------------
def compute_and_store(results, prefix, y_true, y_pred, y_prob):
    """
    Computes seven metrics + 95 % bootstrap BCA CIs and appends them
    (point estimate + lower/upper) into the wide-format results dict.
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
    prec, (prec_lo, prec_hi) = ci_precision_score(
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
    # balanced_accuracy_score is not natively in the library — use generic
    # bootstrap_ci with the sklearn callable directly
    bal, (bal_lo, bal_hi) = bootstrap_ci(
        y_true, y_pred,
        metric=balanced_accuracy_score,
        confidence_level=CONF_LEVEL,
        method=CI_METHOD,
        n_resamples=N_RESAMPLES,
        random_state=42,
    )

    for metric, val, lo, hi in [
        ("accuracy",    acc,  acc_lo,  acc_hi),
        ("f1",          f1,   f1_lo,   f1_hi),
        ("sensitivity", sens, sens_lo, sens_hi),
        ("specificity", spec, spec_lo, spec_hi),
        ("precision",   prec, prec_lo, prec_hi),
        ("auc",         auc,  auc_lo,  auc_hi),
        ("bal_acc",     bal,  bal_lo,  bal_hi),
    ]:
        results[f"{prefix}_{metric}"].append(val)
        results[f"{prefix}_{metric}_ci_lower"].append(lo)
        results[f"{prefix}_{metric}_ci_upper"].append(hi)


# -----------------------------------------------------------------------------
# Reverse LOCO loop
# Train once per country, evaluate on all other countries
# -----------------------------------------------------------------------------
for train_idx, train_row in df.iterrows():

    train_country = train_row["country"]
    print(f"\n{'='*60}")
    print(f"TRAIN: {train_country}")
    print(f"{'='*60}")

    train_dict = load_dhs_kids_data(df, indx=train_idx, variables=VARIABLES)
    train_df   = preprocess_dhs(train_dict["kr"], FEATURES, CAL_COLS, TARGET)

    X_train, y_train, class_weights, train_mean, train_std = \
        prepare_ml_data_loco(train_df, FEATURES, cat_cols, TARGET)

    # ── Fit all models once on this train country ─────────────────────────────

    print("  → Logistic Regression...")
    lg_model = LogisticRegression(max_iter=2000, **best_hyperparams["logreg"])
    lg_model.fit(X_train, y_train)

    print("  → LightGBM...")
    lgbm_model = LGBMClassifier(
        force_row_wise=True,
        max_bin=63,
        n_jobs=4,
        verbosity=-1,
        **best_hyperparams["lgbm"],
    )
    lgbm_model.fit(X_train, y_train)

    print("  → XGBoost...")
    xgb_base = {"eval_metric": "logloss", "random_state": 42, "n_jobs": 4}
    if torch.cuda.is_available():
        xgb_base["device"] = "cuda"
    else:
        xgb_base["tree_method"] = "hist"

    xgb_model = XGBClassifier(**xgb_base, **best_hyperparams["xgb"])
    xgb_model.fit(X_train, y_train)

    print("  → TabPFN...")
    tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
    tabpfn.fit(X_train, y_train)

    # ── Evaluate on every other country ──────────────────────────────────────
    for test_idx in df[df["country"] != train_country].index:

        test_country = df.loc[test_idx, "country"]
        print(f"    Testing on: {test_country}")

        test_dict = load_dhs_kids_data(df, indx=test_idx, variables=VARIABLES)
        test_df   = preprocess_dhs(test_dict["kr"], FEATURES, CAL_COLS, TARGET)

        X_test, y_test, _, _, _ = prepare_ml_data_loco(
            test_df, FEATURES, cat_cols, TARGET,
            train_mean=train_mean,
            train_std=train_std,
        )

        results["train_country"].append(train_country)
        results["test_country"].append(test_country)

        # Logistic Regression
        y_pred = lg_model.predict(X_test)
        y_prob = lg_model.predict_proba(X_test)[:, 1]
        compute_and_store(results, "lg", y_test, y_pred, y_prob)

        # LightGBM
        y_pred = lgbm_model.predict(X_test)
        y_prob = lgbm_model.predict_proba(X_test)[:, 1]
        compute_and_store(results, "lgbm", y_test, y_pred, y_prob)

        # XGBoost
        y_pred = xgb_model.predict(X_test)
        y_prob = xgb_model.predict_proba(X_test)[:, 1]
        compute_and_store(results, "xgb", y_test, y_pred, y_prob)

        # TabPFN
        y_pred = tabpfn.predict(X_test)
        y_prob = tabpfn.predict_proba(X_test)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        compute_and_store(results, "tabpfn", y_test, y_pred, y_prob)

    # ── Save after each train country ─────────────────────────────────────────
    pd.DataFrame(results).to_csv(
        f"{RESULTS_DIR}/anemia_reverse_loco_results.csv", index=False
    )
    print(f"  ✔ Saved after training on {train_country}")

print("\nDONE — Reverse LOCO complete")