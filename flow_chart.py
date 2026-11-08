import numpy as np
import pandas as pd

from config import RESULTS_DIR, VARIABLES
from util import load_dhs_kids_data

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TARGET   = VARIABLES[-1]   # hw57 (anaemia)
FEATURES = VARIABLES[:-1]

AGE_VAR = "hw1"
AGE_MIN, AGE_MAX = 6, 59

DHS_MISSING     = [994, 995, 996, 997, 998, 999]
GENERAL_MISSING = [8, 9, 98, 99, 998, 999, 9998, 9999]
ALL_MISSING     = list(set(DHS_MISSING + GENERAL_MISSING + [-1]))

df = pd.read_csv("./dhs_country_file_summary.csv")

# ─────────────────────────────────────────────
# Tracking
# ─────────────────────────────────────────────
exclusion_records = {
    "country": [],
    "n_raw": [],
    "n_after_target_clean": [],
    "n_excluded_missing_outcome": [],
    "n_after_age": [],
    "n_excluded_age": [],
    "n_final": [],
    "n_missing_predictors": [],
}

# ─────────────────────────────────────────────
# Cohort derivation function (paper-consistent)
# ─────────────────────────────────────────────
def count_exclusions(raw_df, features, target):

    counts = {}
    counts["n_raw"] = len(raw_df)

    df = raw_df.copy()

    # ── Step 1: clean missing codes (target + predictors + age)
    cols = list(set(features + [target, AGE_VAR]))

    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(ALL_MISSING, np.nan)

    if AGE_VAR in df.columns:
        df[AGE_VAR] = pd.to_numeric(df[AGE_VAR], errors="coerce")

    # ─────────────────────────────────────────────
    # Step 2: ensure target availability first
    # ─────────────────────────────────────────────
    before_target = len(df)
    df = df.dropna(subset=[target])
    counts["n_excluded_missing_outcome"] = before_target - len(df)
    counts["n_after_target_clean"] = len(df)

    # ─────────────────────────────────────────────
    # Step 3: age eligibility
    # ─────────────────────────────────────────────
    before_age = len(df)

    if AGE_VAR in df.columns:
        df = df[df[AGE_VAR].between(AGE_MIN, AGE_MAX, inclusive="both")]

    counts["n_excluded_age"] = before_age - len(df)
    counts["n_after_age"] = len(df)

    # ─────────────────────────────────────────────
    # Step 4: final analytic sample
    # ─────────────────────────────────────────────
    df_final = df.copy()
    counts["n_final"] = len(df_final)

    # ─────────────────────────────────────────────
    # Step 5: predictor missingness (pre-imputation)
    # ─────────────────────────────────────────────
    df_feat = df_final[features]

    counts["n_missing_predictors"] = int(
        df_feat.isna().any(axis=1).sum()
    )

    return counts


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
for index, row in df.iterrows():

    country = row["country"]
    print(f"\nProcessing: {country}")

    data_dict = load_dhs_kids_data(df, indx=index, variables=VARIABLES)
    kr_df     = data_dict["kr"]

    excl = count_exclusions(kr_df, FEATURES, TARGET)

    exclusion_records["country"].append(country)
    exclusion_records["n_raw"].append(excl["n_raw"])
    exclusion_records["n_after_target_clean"].append(excl["n_after_target_clean"])
    exclusion_records["n_excluded_missing_outcome"].append(excl["n_excluded_missing_outcome"])
    exclusion_records["n_after_age"].append(excl["n_after_age"])
    exclusion_records["n_excluded_age"].append(excl["n_excluded_age"])
    exclusion_records["n_final"].append(excl["n_final"])
    exclusion_records["n_missing_predictors"].append(excl["n_missing_predictors"])

    print(f"  Raw N:                     {excl['n_raw']:>8,}")
    print(f"  After target clean:        {excl['n_after_target_clean']:>8,}")
    print(f"  Excl missing outcome:      {excl['n_excluded_missing_outcome']:>8,}")
    print(f"  After age filter:          {excl['n_after_age']:>8,}")
    print(f"  Final analytic sample:     {excl['n_final']:>8,}")
    print(f"  Missing predictors:        {excl['n_missing_predictors']:>8,}")


# ─────────────────────────────────────────────
# Save per country
# ─────────────────────────────────────────────
excl_df = pd.DataFrame(exclusion_records)
excl_df.to_csv(f"{RESULTS_DIR}/exclusion_counts_by_country.csv", index=False)


# ─────────────────────────────────────────────
# Pooled flowchart numbers
# ─────────────────────────────────────────────
pooled = {
    "n_raw_total":                      excl_df["n_raw"].sum(),
    "n_after_target_total":            excl_df["n_after_target_clean"].sum(),
    "n_excluded_outcome_total":        excl_df["n_excluded_missing_outcome"].sum(),
    "n_after_age_total":               excl_df["n_after_age"].sum(),
    "n_excluded_age_total":            excl_df["n_excluded_age"].sum(),
    "n_final_total":                   excl_df["n_final"].sum(),
    "n_missing_predictors_total":      excl_df["n_missing_predictors"].sum(),
}

pd.DataFrame([pooled]).to_csv(
    f"{RESULTS_DIR}/exclusion_counts_pooled.csv", index=False
)


# ─────────────────────────────────────────────
# Flowchart output
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FLOWCHART — COHORT DERIVATION")
print("=" * 60)

print(f"  Box 1  Raw total:              {pooled['n_raw_total']:>10,}")
print(f"  Excl 1 Missing outcome:       {pooled['n_excluded_outcome_total']:>10,}")
print(f"  Box 2  After outcome filter:   {pooled['n_after_target_total']:>10,}")
print(f"  Excl 2 Age 6–59 months:       {pooled['n_excluded_age_total']:>10,}")
print(f"  Box 3  Final analytic sample:  {pooled['n_final_total']:>10,}")

print(f"\n  ── Predictor missingness (imputed later) ──")
print(f"  Any missing predictors:       {pooled['n_missing_predictors_total']:>10,}")
print("=" * 60)

print(f"\nOutputs saved to {RESULTS_DIR}/")