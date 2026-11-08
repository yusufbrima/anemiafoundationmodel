
import json
import pandas as pd
import numpy as np
import torch

from pathlib import Path

from config import RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)


# =========================================================
# FILE PATHS
# =========================================================

OUTPUT_FILE = (
    f"{RESULTS_DIR}/subgroup_analysis_optuna_models.csv"
)

AGE_OUTPUT_FILE = (
    f"{RESULTS_DIR}/subgroup_analysis_optuna_models_age_corrected.csv"
)

FINAL_OUTPUT_FILE = (
    f"{RESULTS_DIR}/subgroup_analysis_optuna_models_updated.csv"
)


# =========================================================
# LOAD EXISTING OPTUNA HYPERPARAMETERS
# =========================================================

with open(f"{RESULTS_DIR}/best_hyperparameters.json", "r") as f:
    BEST_PARAMS = json.load(f)

print("Loaded hyperparameters:", BEST_PARAMS.keys())


# =========================================================
# LOAD EXISTING SUBGROUP RESULTS
# =========================================================

if not Path(OUTPUT_FILE).exists():
    raise FileNotFoundError(
        f"Existing results file not found: {OUTPUT_FILE}"
    )

previous_results = pd.read_csv(OUTPUT_FILE)

if "subgroup" not in previous_results.columns:
    raise ValueError(
        "The existing results file does not contain a subgroup column."
    )

# Preserve all existing results except the old age-group results
preserved_results = previous_results[
    previous_results["subgroup"] != "age_group"
].copy()

print("\nExisting results loaded.")
print("Original result rows:", len(previous_results))
print("Preserved non-age rows:", len(preserved_results))
print(
    "Old age-group rows removed:",
    len(previous_results) - len(preserved_results)
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)


# =========================================================
# SETUP
# =========================================================

TARGET = VARIABLES[-1]
FEATURES = VARIABLES[:-1]

df = pd.read_csv("./dhs_country_file_summary.csv")


# =========================================================
# SUBGROUPS
# =========================================================

SUBGROUPS = {
    "sex": "b4",
    "wealth": "v190",
    "education": "v106",
    "residence": "v025",
    "age_group": "age_group"
}


# =========================================================
# AGE GROUP BINNING
# =========================================================

AGE_BINS = [6, 12, 24, 36, 48, 60]

AGE_LABELS = [
    "6–11m",
    "12–23m",
    "24–35m",
    "36–47m",
    "48–59m"
]


# =========================================================
# CV
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# METRICS
# =========================================================

def specificity(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        return np.nan

    tn, fp, fn, tp = cm.ravel()

    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def safe_auc(y_true, y_prob):

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, y_prob)


# =========================================================
# PREPROCESSOR
# =========================================================

def build_preprocessor(X):

    num_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    return ColumnTransformer([

        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ),

        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]),
            cat_cols
        )

    ])


# =========================================================
# MODEL BUILDERS (USING EXISTING OPTUNA PARAMETERS)
# =========================================================

def get_logreg():

    return LogisticRegression(
        max_iter=2000,
        **BEST_PARAMS["logreg"]
    )


def get_lgbm():

    return LGBMClassifier(
        force_row_wise=True,
        max_bin=63,
        n_jobs=4,
        verbosity=-1,
        **BEST_PARAMS["lgbm"]
    )


def get_xgb():

    base = {
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 4,
    }

    if torch.cuda.is_available():

        base["device"] = "cuda"

    else:

        base["tree_method"] = "hist"

    return XGBClassifier(
        **base,
        **BEST_PARAMS["xgb"]
    )


# =========================================================
# STORAGE FOR NEW AGE-GROUP RESULTS
# =========================================================

results_rows = []


# =========================================================
# MAIN LOOP: COUNTRY LEVEL
# =========================================================

for index, row in df.iterrows():

    country = row["country"]

    print("\n" + "=" * 60)
    print(f"Processing country: {country}")
    print("=" * 60)

    # -----------------------------------------------------
    # LOAD COUNTRY DATA
    # -----------------------------------------------------

    data_dict = load_dhs_kids_data(
        df,
        indx=index,
        variables=VARIABLES
    )

    ik_df = data_dict["kr"]

    df_clean = preprocess_dhs(
        ik_df,
        FEATURES,
        CAL_COLS,
        TARGET
    )

    # -----------------------------------------------------
    # CORRECT AGE GROUP BINNING
    # -----------------------------------------------------

    if "hw1" not in df_clean.columns:

        print(
            f"WARNING: hw1 not found for {country}. Skipping."
        )

        continue

    df_clean["hw1"] = pd.to_numeric(
        df_clean["hw1"],
        errors="coerce"
    )

    valid_ages = df_clean["hw1"].dropna()

    if valid_ages.empty:

        print(
            f"WARNING: No valid age observations for {country}. "
            "Skipping."
        )

        continue

    assert valid_ages.min() >= 6, (
        f"preprocess_dhs should have filtered age. "
        f"Minimum age in {country}: {valid_ages.min()}"
    )

    df_clean["age_group"] = pd.cut(
        df_clean["hw1"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False
    )

    # -----------------------------------------------------
    # CONVERT DEMOGRAPHIC VARIABLES
    # -----------------------------------------------------

    for col in ["b4", "v190", "v106", "v025"]:

        if col in df_clean.columns:

            df_clean[col] = df_clean[col].astype(str)

    # -----------------------------------------------------
    # AGE GROUP DISTRIBUTION
    # -----------------------------------------------------

    print("\nCorrected age-group counts:")

    print(
        df_clean["age_group"].value_counts().reindex(
            AGE_LABELS,
            fill_value=0
        )
    )


    # =====================================================
    # SUBGROUP LOOP: AGE ONLY
    # =====================================================

    for subgroup_name, subgroup_col in SUBGROUPS.items():

        # Do not rerun sex, wealth, education or residence
        if subgroup_name != "age_group":

            continue

        if subgroup_col not in df_clean.columns:

            continue

        values = df_clean[subgroup_col].dropna().unique()

        # -------------------------------------------------
        # INDIVIDUAL AGE GROUPS
        # -------------------------------------------------

        for val in values:

            print("\n----------------------------------------")
            print(f"Country: {country}")
            print(f"Subgroup: {subgroup_name}")
            print(f"Age group: {val}")
            print("----------------------------------------")

            sub_df = df_clean[
                df_clean[subgroup_col] == val
            ].copy()

            # Same complete-case filtering as original analysis
            sub_df = sub_df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            sub_df = sub_df.dropna(
                subset=[TARGET] + FEATURES + CAL_COLS
            )

            n_samples = len(sub_df)

            print("Complete-case sample size:", n_samples)

            # -------------------------------------------------
            # MINIMUM SAMPLE SIZE
            # -------------------------------------------------

            if n_samples < 50:

                print(
                    f"Skipping {val}: n={n_samples} < 50"
                )

                continue

            # -------------------------------------------------
            # TRAIN / TEST SPLIT
            # -------------------------------------------------

            X_train, X_test, y_train, y_test, _ = prepare_ml_data(
                sub_df,
                FEATURES,
                CAL_COLS,
                TARGET
            )

            if len(np.unique(y_train)) < 2:

                print(
                    f"Skipping {val}: only one outcome class "
                    "in training data."
                )

                continue

            # -------------------------------------------------
            # PREPROCESSING
            # -------------------------------------------------

            pre = build_preprocessor(X_train)

            X_train_p = pre.fit_transform(X_train)

            X_test_p = pre.transform(X_test)


            # =================================================
            # LOGISTIC REGRESSION
            # =================================================

            print("Training Logistic Regression...")

            lg = get_logreg()

            lg.fit(
                X_train_p,
                y_train
            )

            pred = lg.predict(X_test_p)

            prob = lg.predict_proba(X_test_p)[:, 1]

            results_rows.append({

                "country": country,
                "subgroup": subgroup_name,
                "value": str(val),
                "model": "LogReg",

                "accuracy": accuracy_score(y_test, pred),

                "f1": f1_score(y_test, pred),

                "sensitivity": recall_score(y_test, pred),

                "specificity": specificity(y_test, pred),

                "auc": safe_auc(y_test, prob),

            })


            # =================================================
            # LIGHTGBM
            # =================================================

            print("Training LightGBM...")

            lgbm = get_lgbm()

            lgbm.fit(
                X_train_p,
                y_train
            )

            pred = lgbm.predict(X_test_p)

            prob = lgbm.predict_proba(X_test_p)[:, 1]

            results_rows.append({

                "country": country,
                "subgroup": subgroup_name,
                "value": str(val),
                "model": "LightGBM",

                "accuracy": accuracy_score(y_test, pred),

                "f1": f1_score(y_test, pred),

                "sensitivity": recall_score(y_test, pred),

                "specificity": specificity(y_test, pred),

                "auc": safe_auc(y_test, prob),

            })


            # =================================================
            # XGBOOST
            # =================================================

            print("Training XGBoost...")

            xgb = get_xgb()

            xgb.fit(
                X_train_p,
                y_train
            )

            pred = xgb.predict(X_test_p)

            prob = xgb.predict_proba(X_test_p)[:, 1]

            results_rows.append({

                "country": country,
                "subgroup": subgroup_name,
                "value": str(val),
                "model": "XGBoost",

                "accuracy": accuracy_score(y_test, pred),

                "f1": f1_score(y_test, pred),

                "sensitivity": recall_score(y_test, pred),

                "specificity": specificity(y_test, pred),

                "auc": safe_auc(y_test, prob),

            })


            # =================================================
            # TABPFN
            # =================================================

            print("Training TabPFN...")

            tabpfn = TabPFNClassifier(
                ignore_pretraining_limits=True
            )

            tabpfn.fit(
                X_train_p,
                y_train
            )

            pred = tabpfn.predict(X_test_p)

            prob = tabpfn.predict_proba(X_test_p)[:, 1]

            results_rows.append({

                "country": country,
                "subgroup": subgroup_name,
                "value": str(val),
                "model": "TabPFN",

                "accuracy": accuracy_score(y_test, pred),

                "f1": f1_score(y_test, pred),

                "sensitivity": recall_score(y_test, pred),

                "specificity": specificity(y_test, pred),

                "auc": safe_auc(y_test, prob),

            })


# =========================================================
# SAVE CORRECTED AGE-GROUP RESULTS
# =========================================================

age_results = pd.DataFrame(results_rows)

if age_results.empty:

    raise RuntimeError(
        "No corrected age-group results were generated. "
        "The original results file has not been modified."
    )

age_results.to_csv(
    AGE_OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("CORRECTED AGE-GROUP ANALYSIS COMPLETED")
print("=" * 60)

print("Number of corrected age-group result rows:", len(age_results))

print("\nCorrected age-group distribution:")

print(
    age_results.groupby(
        ["country", "value"]
    )["model"].count()
)

print("\nSaved corrected age results to:")
print(AGE_OUTPUT_FILE)


# =========================================================
# MERGE WITH EXISTING NON-AGE RESULTS
# =========================================================

merged_results = pd.concat(
    [
        preserved_results,
        age_results
    ],
    ignore_index=True
)


# =========================================================
# VALIDATION BEFORE SAVING
# =========================================================

# Confirm that only age-group results were recalculated
assert set(age_results["subgroup"].unique()) == {"age_group"}

# Confirm that all original non-age rows were preserved
assert (
    len(merged_results) ==
    len(preserved_results) + len(age_results)
)

# Confirm that the merged data contain all expected subgroups
expected_subgroups = {
    "sex",
    "wealth",
    "education",
    "residence",
    "age_group"
}

actual_subgroups = set(
    merged_results["subgroup"].dropna().unique()
)

assert expected_subgroups.issubset(actual_subgroups), (
    f"Missing subgroups: {expected_subgroups - actual_subgroups}"
)


# =========================================================
# SAVE FINAL MERGED RESULTS
# =========================================================

merged_results.to_csv(
    FINAL_OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("FINAL RESULTS MERGED SUCCESSFULLY")
print("=" * 60)

print("\nFinal results saved to:")
print(FINAL_OUTPUT_FILE)

print("\nTotal result rows:", len(merged_results))

print("\nResults per subgroup:")

print(
    merged_results.groupby("subgroup").size()
)

print("\nNumber of countries per subgroup:")

print(
    merged_results.groupby("subgroup")["country"].nunique()
)

print("\nModels per age group:")

print(
    age_results.groupby(
        ["value", "model"]
    ).size()
)

print("\nDONE")
