from __future__ import annotations
from typing import Literal
import pandas as pd
import os
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required to unlock IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.utils.class_weight import compute_class_weight
import importlib
import subprocess
import sys


# ─────────────────────────────────────────────
# Missing codes — single source of truth
# kept in sync with exclusion_tracking.py
# and prepare_ml_data
# ─────────────────────────────────────────────
DHS_MISSING     = [994, 995, 996, 997, 998, 999]
GENERAL_MISSING = [8, 9, 98, 99, 998, 999, 9998, 9999]
ALL_MISSING     = list(set(DHS_MISSING + GENERAL_MISSING + [-1]))
 
AGE_VAR = "hw1"
AGE_MIN = 6
AGE_MAX = 59





def install_if_missing(package_name, import_name=None):
    """
    package_name: name used in pip install
    import_name: name used in import (if different)
    """
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        print(f"{package_name} is already installed ✅")
    except ImportError:
        print(f"{package_name} not found. Installing...")
        # if import_name == 'lightgbm':
        #     subprocess.check_call([sys.executable, "-m", "pip", "install", 'lightgbm --no-binary lightgbm'])
        # else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} installed successfully ✅")


# Install packages safely
install_if_missing("lightgbm")
install_if_missing("tabpfn")
install_if_missing("xgboost")
install_if_missing("tabpfn_client")
install_if_missing("pyreadstat") 
install_if_missing("optuna") 
install_if_missing("tabicl") 
install_if_missing("joblib") 
install_if_missing("cupy-cuda12x") 
install_if_missing("dcurves") 
install_if_missing("statkit") 
install_if_missing("python-dotenv") 
install_if_missing("tabpfn-extensions[interpretability] @ git+https://github.com/PriorLabs/tabpfn-extensions.git") 


def load_dhs_kids_data(df, indx, variables, max_rows=24000):
    """
    Load DHS Kids Recode (KR) Stata file for a given index.

    Parameters:
        df (pd.DataFrame): DataFrame containing file paths and metadata,
                           must include 'country' and 'kr_files' columns.
        indx (int): Index of the row to process.
        variables (list): List of columns to read
        max_rows (int): Maximum number of rows to load

    Returns:
        dict: Dictionary with key "kr" mapping to the loaded DataFrame,
              or an empty dict if loading fails.
    """
    import pandas as pd

    row = df.iloc[indx]
    print(f"Index: {indx} | Country: {row['country']}")

    data = {}
    try:
        file_path = row['kr_files']

        chunks = []
        total = 0

        # Read in chunks
        reader = pd.read_stata(
            file_path,
            convert_categoricals=True,
            columns=variables,
            chunksize=5000
        )

        for chunk in reader:
            chunks.append(chunk)
            total += len(chunk)

            if total >= max_rows:
                break

        # Combine and trim to exact max_rows
        kr_stata_df = pd.concat(chunks).head(max_rows)

        print(f"Kids Recode Shape: {kr_stata_df.shape}")
        data["kr"] = kr_stata_df

    except FileNotFoundError:
        print(f"[ERROR] KR file not found for {row['country']}: {row['kr_files']}")
    except Exception as e:
        print(f"[ERROR] Failed to load KR file for {row['country']}: {e}")

    return data


def preprocess_dhs(df, features, cat_cols, target):
    """
    DHS preprocessing pipeline — consistent with exclusion_tracking.py.
 
    Steps
    -----
    1. Replace all missing codes with NaN (features + target + age).
    2. Coerce age to numeric.
    3. Apply age eligibility filter (6–59 months, hw1).
    4. Convert categorical columns.
    5. Drop rows with missing target (hw57).
 
    Missing codes used
    ------------------
    DHS standard : 994–999
    General      : 8, 9, 98, 99, 998, 999, 9998, 9999
    Sentinel     : -1
    """
 
    df_processed = df.copy()
 
    # ── 1. Replace missing codes ─────────────
    # clean features + target + age in one pass
    cols_to_clean = list(set(features + [target, AGE_VAR]))
 
    for col in cols_to_clean:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace(ALL_MISSING, np.nan)
 
    # ── 2. Coerce age to numeric ─────────────
    if AGE_VAR in df_processed.columns:
        df_processed[AGE_VAR] = pd.to_numeric(
            df_processed[AGE_VAR], errors="coerce"
        )
    else:
        print(f"WARNING: {AGE_VAR} not found — age filter will be skipped")
 
    # ── 3. Age filter (WHO: 6–59 months) ─────
    if AGE_VAR in df_processed.columns:
        before_age = len(df_processed)
        df_processed = df_processed[
            df_processed[AGE_VAR].between(AGE_MIN, AGE_MAX, inclusive="both")
        ]
        print(
            f"Age filter ({AGE_VAR} {AGE_MIN}–{AGE_MAX}m): "
            f"{len(df_processed):,}  (−{before_age - len(df_processed):,})"
        )
 
    # ── 4. Categorical columns ───────────────
    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype("category")
 
    # ── 5. Drop missing target ───────────────
    if target in df_processed.columns:
        before_target = len(df_processed)
        df_processed = df_processed.dropna(subset=[target])
        print(
            f"Drop missing target ({target}): "
            f"{len(df_processed):,}  (−{before_target - len(df_processed):,})"
        )
    else:
        print(f"WARNING: target '{target}' not found — no rows dropped")
 
    print(f"Final shape: {df_processed.shape}")
 
    return df_processed
 



def prepare_ml_data_loco(
    df,
    features,
    cat_cols,
    target,
    train_mean=None,
    train_std=None
):
    """
    LOCO-compatible preprocessing:
    - NO train/test split inside
    - works for both train and test sets
    - scaling controlled externally
    """

    df_proc = df.copy()

    # -----------------------------
    # 1. Target (binary anemia)
    # -----------------------------
    y = df_proc[target].astype(str).str.lower().isin(
        ["mild", "moderate", "severe"]
    ).astype(int)

    # -----------------------------
    # 2. Feature matrix
    # -----------------------------
    X = df_proc[features].copy()

    # -----------------------------
    # 3. Missing values
    # -----------------------------
    dhs_missing = [994, 995, 996, 997, 998, 999, -1]
    X = X.replace(dhs_missing, np.nan)

    # -----------------------------
    # 4. Column types
    # -----------------------------
    num_cols = [c for c in X.columns if c not in cat_cols]

    # -----------------------------
    # 5. Numeric processing
    # -----------------------------
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # -----------------------------
    # 6. Categorical imputation
    # -----------------------------
    for col in cat_cols:
        if col in X.columns:
            mode_val = X[col].mode(dropna=True)
            if len(mode_val) > 0:
                X[col] = X[col].fillna(mode_val.iloc[0])

    # -----------------------------
    # 7. Encoding
    # -----------------------------
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes

    # -----------------------------
    # 8. Scaling (LOCO-safe)
    # -----------------------------
    if train_mean is None or train_std is None:
        train_mean = X[num_cols].mean()
        train_std = X[num_cols].std()

    X[num_cols] = (X[num_cols] - train_mean) / train_std

    # -----------------------------
    # 9. Class weights
    # -----------------------------
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    class_weights = {0: weights[0], 1: weights[1]}

    return X, y, class_weights, train_mean, train_std


def prepare_ml_data(
    df,
    features,
    cat_cols,
    target,
    test_size=0.2,
    imputation_method="median_mode",   # "median_mode" (default), "mice", or "knn"
    mice_max_iter=10,
    mice_random_state=42,
    knn_n_neighbors=5,
):
    """
    Full ML preprocessing pipeline:
    - Binary target creation (anemia)
    - Missing value handling
    - Categorical encoding
    - Train/test split
    - Imputation + scaling

    Parameters
    ----------
    df : pd.DataFrame
    features : list
    cat_cols : list
    target : str
    test_size : float
    imputation_method : str, default "median_mode"
        "median_mode" -> median imputation for numeric, mode imputation for categorical.
        "mice"        -> single-imputation iterative chained-equations ("MICE-lite"),
                          via sklearn's IterativeImputer, fit on training data only.
        "knn"         -> k-nearest-neighbors imputation via sklearn's KNNImputer,
                          fit on training data only.
    mice_max_iter : int
        Number of chained-equations iterations used when imputation_method="mice".
    mice_random_state : int
        Random state for IterativeImputer reproducibility.
    knn_n_neighbors : int
        Number of neighbors used when imputation_method="knn".

    Returns
    -------
    X_train, X_test, y_train, y_test, class_weights
    """

    assert imputation_method in ("median_mode", "mice", "knn"), \
        f"imputation_method must be 'median_mode', 'mice', or 'knn', got {imputation_method!r}"

    df_proc = df.copy()

    # -----------------------------
    # 1. Target (binary anemia)
    # -----------------------------
    Y = df_proc[target].astype(str).str.lower().isin(
        ["mild", "moderate", "severe"]
    ).astype(int)

    # -----------------------------
    # 2. Feature matrix
    # -----------------------------
    X = df_proc[features].copy()

    # -----------------------------
    # 3. Missing values (DHS missing codes -> NaN)
    # -----------------------------
    dhs_missing = [994, 995, 996, 997, 998, 999, -1]
    X = X.replace(dhs_missing, np.nan)

    # -----------------------------
    # 4. Column types
    # -----------------------------
    num_cols = [c for c in X.columns if c not in cat_cols]

    # -----------------------------
    # 5. Numeric coercion (no imputation yet — leakage-safe)
    # -----------------------------
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")

    # -----------------------------
    # 6. Categorical encoding (NaN preserved, not filled yet)
    # -----------------------------
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes.astype(float)
            X.loc[X[col] == -1, col] = np.nan  # cat.codes uses -1 for NaN; restore as NaN

    remaining_nan = X.isna().mean().sort_values(ascending=False)
    print("remaining_nan (pre-split, pre-imputation): ", remaining_nan.head(20))

    # -----------------------------
    # 7. Train/test split (BEFORE any imputation, to prevent leakage)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=42,
        stratify=Y
    )

    # -----------------------------
    # 8. Imputation (fit on train only, applied to test)
    # -----------------------------
    if imputation_method == "median_mode":
        num_imputer = SimpleImputer(strategy="median")
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

        for col in cat_cols:
            if col in X.columns:
                mode_val = X_train[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                X_train[col] = X_train[col].fillna(fill_val)
                X_test[col] = X_test[col].fillna(fill_val)

    elif imputation_method == "mice":
        mice_imputer = IterativeImputer(
            max_iter=mice_max_iter,
            sample_posterior=False,   # single imputation, not multiple draws
            keep_empty_features=True,   # prevents column drop when a fold has an all-NaN feature
            random_state=mice_random_state,
        )
        X_train_arr = mice_imputer.fit_transform(X_train)
        X_test_arr = mice_imputer.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_arr, columns=X.columns, index=X_test.index)

        # Round imputed categorical columns back to valid integer category codes
        for col in cat_cols:
            if col in X.columns:
                max_code = X[col].max()
                X_train[col] = X_train[col].round().clip(0, max_code)
                X_test[col] = X_test[col].round().clip(0, max_code)

    elif imputation_method == "knn":
        # KNNImputer requires scaled inputs for distances to be meaningful across
        # features of different units; we standardize numeric columns using
        # train-only statistics purely for the neighbor-search step, impute,
        # then un-scale before returning (so downstream standardization in
        # step 10 behaves identically across all imputation_method branches).
        knn_train_mean = X_train[num_cols].mean()
        knn_train_std = X_train[num_cols].std().replace(0, 1.0)

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[num_cols] = (X_train[num_cols] - knn_train_mean) / knn_train_std
        X_test_scaled[num_cols] = (X_test[num_cols] - knn_train_mean) / knn_train_std

        knn_imputer = KNNImputer(n_neighbors=knn_n_neighbors, keep_empty_features=True)
        X_train_arr = knn_imputer.fit_transform(X_train_scaled)
        X_test_arr = knn_imputer.transform(X_test_scaled)

        X_train = pd.DataFrame(X_train_arr, columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_arr, columns=X.columns, index=X_test.index)

        # Un-scale numeric columns back to raw units
        X_train[num_cols] = X_train[num_cols] * knn_train_std + knn_train_mean
        X_test[num_cols] = X_test[num_cols] * knn_train_std + knn_train_mean

        # Round imputed categorical columns back to valid integer category codes
        for col in cat_cols:
            if col in X.columns:
                max_code = X[col].max()
                X_train[col] = X_train[col].round().clip(0, max_code)
                X_test[col] = X_test[col].round().clip(0, max_code)

    # -----------------------------
    # 9. Final check
    # -----------------------------
    assert X_train.isna().sum().sum() == 0, "Still have NaNs in X_train!"
    assert X_test.isna().sum().sum() == 0, "Still have NaNs in X_test!"

    # -----------------------------
    # 10. Standardization (numeric only, train stats applied to both)
    # -----------------------------
    train_mean = X_train[num_cols].mean()
    train_std = X_train[num_cols].std()

    X_train[num_cols] = (X_train[num_cols] - train_mean) / train_std
    X_test[num_cols] = (X_test[num_cols] - train_mean) / train_std

    # -----------------------------
    # 11. Class weights
    # -----------------------------
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = {0: weights[0], 1: weights[1]}
    print(class_weights)

    return X_train, X_test, y_train, y_test, class_weights



if __name__ == "__main__":
    pass