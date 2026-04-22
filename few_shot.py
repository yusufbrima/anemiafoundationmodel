import pandas as pd
import numpy as np
from pathlib import Path
import json
from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

import torch

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Setup
# --------------------------------------------------
TARGET = VARIABLES[-1]
FEATURES = VARIABLES[:-1]

output_file = "./dhs_country_file_summary.csv"
df = pd.read_csv(output_file)

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Few-shot levels
# Example:
# 0.20 = use 20% train, 80% test
# --------------------------------------------------
# SHOT_LEVELS = [0.01, 0.05, 0.10, 0.20]
SHOT_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)



with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    best_hyperparams = json.load(f)


# --------------------------------------------------
# Loop over countries
# --------------------------------------------------
for index, row in df.iterrows():

    country = row["country"]
    print(f"\nProcessing: {country}")

    data_dict = load_dhs_kids_data(
        df,
        indx=index,
        variables=VARIABLES
    )

    ik_stata_df = data_dict["kr"]

    df_clean = preprocess_dhs(
        ik_stata_df,
        FEATURES,
        CAL_COLS,
        TARGET
    )

    n_total = len(df_clean)
    print("Final df size =", n_total)

    country_results = []

    # ----------------------------------------------
    # Only inner loop = shot sizes
    # ----------------------------------------------
    for shot in SHOT_LEVELS:

        # ------------------------------------------
        # Stratified split
        # ------------------------------------------
        X_train, X_test, y_train, y_test, class_weights = prepare_ml_data(
            df_clean,
            FEATURES,
            CAL_COLS,
            TARGET,
            test_size=1 - shot
        )
        n_shot = len(X_train)
        n_test = len(X_test)

        print(
            f"shot={shot:.2f} | "
            f"n_shot={n_shot} | "
            f"n_test={n_test}"
        )

        # skip impossible runs
        if n_shot < 20 or n_test < 20:
            continue
        
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        # ==================================================
        # 1. Logistic Regression
        # ==================================================
        logreg = LogisticRegression(max_iter=1000,**best_hyperparams["logreg"])



        logreg.fit(X_train, y_train)
        # best_lg = lg_search.best_estimator_

        y_pred = logreg.predict(X_test)
        y_prob = logreg.predict_proba(X_test)[:, 1]

        country_results.append({
            "country": country,
            "shot": shot,
            "n_shot": len(X_train),
            "n_test": len(X_test),
            "model": "LogReg",
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "sensitivity": recall_score(y_test, y_pred),
            "specificity": specificity(y_test, y_pred),
            "auc": safe_auc(y_test, y_prob)
        })

        # ==================================================
        # 2. LightGBM
        # ==================================================
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

        country_results.append({
            "country": country,
            "shot": shot,
            "n_shot": len(X_train),
            "n_test": len(X_test),
            "model": "LightGBM",
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "sensitivity": recall_score(y_test, y_pred),
            "specificity": specificity(y_test, y_pred),
            "auc": safe_auc(y_test, y_prob)
        })

        # ==================================================
        # 3. XGBoost
        # ==================================================
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

        y_pred = xgb.predict(X_test)
        y_prob = xgb.predict_proba(X_test)[:, 1]

        country_results.append({
            "country": country,
            "shot": shot,
            "n_shot": len(X_train),
            "n_test": len(X_test),
            "model": "XGBoost",
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "sensitivity": recall_score(y_test, y_pred),
            "specificity": specificity(y_test, y_pred),
            "auc": safe_auc(y_test, y_prob)
        })

        # ==================================================
        # 4. TabPFN
        # ==================================================
        print("Training TabPFN...")

        tabpfn = TabPFNClassifier(
            ignore_pretraining_limits=True
        )

        tabpfn.fit(X_train, y_train)

        y_pred = tabpfn.predict(X_test)
        y_prob = tabpfn.predict_proba(X_test)

        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        country_results.append({
            "country": country,
            "shot": shot,
            "n_shot": len(X_train),
            "n_test": len(X_test),
            "model": "TabPFN",
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "sensitivity": recall_score(y_test, y_pred),
            "specificity": specificity(y_test, y_pred),
            "auc": safe_auc(y_test, y_prob)
        })

        # --------------------------------------------------
        # Save each country separately
        # --------------------------------------------------
        result_df = pd.DataFrame(country_results)

        save_name = country.lower().replace(" ", "_")
        save_file = f"{RESULTS_DIR}/{save_name}_fewshot_results.csv"

        result_df.to_csv(save_file, index=False)

    print("Saved:", save_file)

print("\nDONE")