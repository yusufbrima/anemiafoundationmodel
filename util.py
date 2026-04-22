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
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import importlib
import subprocess
import sys





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


# def load_dhs_kids_data(df, indx,variables):
#     """
#     Load DHS Kids Recode (KR) Stata file for a given index.

#     Parameters:
#         df (pd.DataFrame): DataFrame containing file paths and metadata,
#                            must include 'country' and 'kr_files' columns.
#         indx (int): Index of the row to process.
#         variables (list): List of columns to read

#     Returns:
#         dict: Dictionary with key "kr" mapping to the loaded DataFrame,
#               or an empty dict if loading fails.
#     """
#     import pyreadstat

#     row = df.iloc[indx]
#     print(f"Index: {indx} | Country: {row['country']}")

#     data = {}
#     try:
#         # kr_stata_df, _ = pyreadstat.read_dta(row['kr_files'])
#         kr_stata_df = pd.read_stata(row['kr_files'],convert_categoricals=True)

#         print(f"Kids Recode Shape: {kr_stata_df.shape}")
#         data["kr"] = kr_stata_df

#     except FileNotFoundError:
#         print(f"[ERROR] KR file not found for {row['country']}: {row['kr_files']}")
#     except Exception as e:
#         print(f"[ERROR] Failed to load KR file for {row['country']}: {e}")

#     return data



def preprocess_dhs(df, features, cat_cols, target):
    """
    Basic DHS preprocessing:
    - Replace missing codes with NaN
    - Convert categorical variables
    - Drop rows with missing target

    Parameters
    ----------
    df : pd.DataFrame
    features : list
    target : str

    Returns
    -------
    pd.DataFrame
    """

    df_processed = df.copy()

    # DHS missing codes
    dhs_missing = [994, 995, 996, 997, 998, 999]
    general_missing = [8, 9, 98, 99, 998, 999, 9998, 9999]

    # # Categorical columns (fixed from your original code)
    # cat_cols = [
    #     "h22", "h11", "h43", "h34", "v190",
    #     "v113", "v116", "v106", "h31",
    #     "v025", "b4", "v151", "v463z"
    # ]

    # Replace DHS missing values
    for col in features:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace(dhs_missing, np.nan)

    # Replace general missing values (features + target)
    all_cols = features + [target]
    for col in all_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace(general_missing, np.nan)

    # Convert categorical columns
    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype("category")

    print("Shape before dropping NA:", df_processed.shape)

    # Drop rows with missing target
    if target in df_processed.columns:
        df_processed = df_processed.dropna(subset=[target])

    print("Shape after dropping NA:", df_processed.shape)

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



def prepare_ml_data(df, features,cat_cols, target,test_size=0.2):
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
    target : str

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

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
    # 8. Final check
    # -----------------------------
    assert X.isna().sum().sum() == 0, "Still have NaNs in X!"

    # -----------------------------
    # 9. Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=42,
        stratify=Y
    )

    # -----------------------------
    # 10. Imputation (safety)
    # -----------------------------
    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    # -----------------------------
    # 11. Standardization
    # -----------------------------
    train_mean = X_train[num_cols].mean()
    train_std = X_train[num_cols].std()

    X_train[num_cols] = (X_train[num_cols] - train_mean) / train_std
    X_test[num_cols] = (X_test[num_cols] - train_mean) / train_std

    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {0: weights[0], 1: weights[1]}
    print(class_weights)

    return X_train, X_test, y_train, y_test,class_weights



if __name__ == "__main__":
    pass