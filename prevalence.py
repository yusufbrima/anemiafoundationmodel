import pandas as pd
import numpy as np
import json
import os
import torch
import shap

from config import DATA_ROOT_DIR, FIGURES_DIR, RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data


# -----------------------------------------------------------------------------
# Load data and hyperparameters
# -----------------------------------------------------------------------------
df = pd.read_csv("./dhs_country_file_summary.csv")

TARGET   = VARIABLES[-1]
FEATURES = VARIABLES[:-1]
# -----------------------------------------------------------------------------
# Main loop — iterate over countries
# -----------------------------------------------------------------------------

prevalence_results = []

for index, row in df.iterrows():

    country = row["country"]

    print(f"\n{'='*60}")
    print(f"Processing: {country}")
    print(f"{'='*60}")

    data_dict = load_dhs_kids_data(
        df,
        indx=index,
        variables=VARIABLES
    )

    df_clean = preprocess_dhs(
        data_dict["kr"],
        FEATURES,
        CAL_COLS,
        TARGET
    )

    X_train, X_test, y_train, y_test, class_weights = prepare_ml_data(
        df_clean,
        FEATURES,
        CAL_COLS,
        TARGET
    )

    # Overall prevalence
    y_all = df_clean[TARGET].astype(str).str.lower().isin(
        ["mild", "moderate", "severe"]
    ).astype(int)

    prevalence_results.append({
        "Country": country,

        "N_total": len(y_all),
        "Overall_prevalence_%": round(y_all.mean() * 100, 2),

        "N_train": len(y_train),
        "Train_prevalence_%": round(y_train.mean() * 100, 2),

        "N_test": len(y_test),
        "Test_prevalence_%": round(y_test.mean() * 100, 2)
    })
    if country == "Bolivia_DHS_2008":
        break

# ------------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------------

prev_df = pd.DataFrame(prevalence_results)

prev_df.to_csv(
     f"{RESULTS_DIR}/country_anemia_prevalence.csv",
    index=False
)

print(prev_df)