"""
cohort_flow_and_missingness.py

Extends your existing exclusion-tracking pattern (same loader, same loop
structure) to explicitly reproduce the manuscript's stated cohort-derivation
order:

    raw records
      -> drop missing target (hw57)      [manuscript: 80,493 excluded]
      -> drop age outside 6-59 months    [manuscript: 2,881 excluded]
      -> final analytic cohort           [manuscript: 68,856]
      -> missingness on FEATURES only, single shared denominator = final N

preprocess_dhs() applies these two filters in the opposite order (age filter
before target-drop), so its intermediate counts won't match the manuscript's
80,493 / 2,881 split even though the final N should agree. This script
applies the filters directly, in the manuscript's order, so the printed
per-stage counts are directly comparable to the flowchart text.

Run:
    python cohort_flow_and_missingness.py
"""

import os
import pandas as pd
import numpy as np

from config import RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data

# These must match whatever preprocess_dhs uses internally for missing codes
# and age eligibility. Adjust the import if they live elsewhere.
try:
    from util import ALL_MISSING, AGE_VAR, AGE_MIN, AGE_MAX
except ImportError:
    raise ImportError(
        "Could not import ALL_MISSING / AGE_VAR / AGE_MIN / AGE_MAX from util. "
        "Check preprocess_dhs() in util.py for where these are actually "
        "defined and update this import (or hardcode the same values here)."
    )

TARGET = VARIABLES[-1]
FEATURES = VARIABLES[:-1]

df = pd.read_csv("./dhs_country_file_summary.csv")

country_flow = []      # per-country cohort-flow rows
missingness_frames = []  # per-country cleaned frames, concatenated at the end

totals = {"raw": 0, "after_target_drop": 0, "final": 0,
          "excluded_target": 0, "excluded_age": 0}

for index, row in df.iterrows():

    country = row["country"]
    print(f"Processing: {country}")

    data_dict = load_dhs_kids_data(df, indx=index, variables=VARIABLES)
    raw_df = data_dict["kr"].copy()
    raw_total = len(raw_df)

    # -----------------------------
    # Replace missing codes (features + target + age), coerce age numeric
    # -----------------------------
    cols_to_clean = list(set(FEATURES + [TARGET, AGE_VAR]))
    for col in cols_to_clean:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].replace(ALL_MISSING, np.nan)

    if AGE_VAR in raw_df.columns:
        raw_df[AGE_VAR] = pd.to_numeric(raw_df[AGE_VAR], errors="coerce")
    else:
        print(f"  WARNING: {AGE_VAR} not found for {country}, skipping")
        continue

    # -----------------------------
    # Stage 1: drop missing target (manuscript order: FIRST)
    # -----------------------------
    step1 = raw_df.dropna(subset=[TARGET])
    excluded_target = raw_total - len(step1)

    # -----------------------------
    # Stage 2: age eligibility filter (manuscript order: SECOND)
    # -----------------------------
    step2 = step1[step1[AGE_VAR].between(AGE_MIN, AGE_MAX, inclusive="both")]
    excluded_age = len(step1) - len(step2)

    final_total = len(step2)

    # -----------------------------
    # Categorical dtype conversion (same as preprocess_dhs step 4)
    # -----------------------------
    clean_df = step2.copy()
    for col in CAL_COLS:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype("category")

    print(
        f"  raw={raw_total:,}  -target={excluded_target:,}  "
        f"-age={excluded_age:,}  final={final_total:,}"
    )

    country_flow.append({
        "country": country,
        "raw_total": raw_total,
        "excluded_missing_target": excluded_target,
        "excluded_age_ineligible": excluded_age,
        "final_total": final_total,
    })

    totals["raw"] += raw_total
    totals["excluded_target"] += excluded_target
    totals["excluded_age"] += excluded_age
    totals["final"] += final_total

    missingness_frames.append(clean_df)

# -----------------------------------------------------------------------------
# Save per-country cohort flow
# -----------------------------------------------------------------------------
flow_df = pd.DataFrame(country_flow)
flow_path = os.path.join(RESULTS_DIR, "cohort_flow_by_country.csv")
flow_df.to_csv(flow_path, index=False)
print(f"\nSaved per-country cohort flow -> {flow_path}")

# -----------------------------------------------------------------------------
# Pooled cohort-flow reconciliation vs manuscript numbers
# -----------------------------------------------------------------------------
print("\n--- Pooled reconciliation vs manuscript-reported numbers ---")
checks = [
    ("Raw total", totals["raw"], 152230),
    ("Excluded: missing target", totals["excluded_target"], 80493),
    ("Excluded: age ineligible", totals["excluded_age"], 2881),
    ("Final analytic cohort", totals["final"], 68856),
]
for label, computed, expected in checks:
    flag = "OK" if computed == expected else "MISMATCH"
    print(f"  {label:30s} computed={computed:>8,}  expected={expected:>8,}  [{flag}]")

# -----------------------------------------------------------------------------
# Missingness table: FEATURES only, single shared denominator = final N
# -----------------------------------------------------------------------------
df_final_all = pd.concat(missingness_frames, ignore_index=True)
N_FINAL = len(df_final_all)

present_features = [c for c in FEATURES if c in df_final_all.columns]
missing_features = [c for c in FEATURES if c not in df_final_all.columns]
if missing_features:
    print(f"\nWARNING: predictors absent from combined data, excluded from "
          f"table: {missing_features}")

miss_mask = df_final_all[present_features].isna()

pooled_missingness = pd.DataFrame({
    "variable": present_features,
    "n_total": N_FINAL,
    "n_missing": miss_mask.sum().values,
    "pct_missing": (miss_mask.mean() * 100).round(2).values,
}).sort_values("pct_missing", ascending=False).reset_index(drop=True)

miss_path = os.path.join(RESULTS_DIR, "missingness_pooled_reconciled.csv")
pooled_missingness.to_csv(miss_path, index=False)
print(f"\nSaved reconciled missingness table (N={N_FINAL:,}) -> {miss_path}")
print(pooled_missingness.to_string(index=False))

# -----------------------------------------------------------------------------
# Rows with >=1 missing predictor, scoped to FEATURES only
# -----------------------------------------------------------------------------
rows_any_missing = miss_mask.any(axis=1).sum()
pct_any_missing = rows_any_missing / N_FINAL * 100
print(
    f"\nRows with >=1 missing predictor (FEATURES only): "
    f"{rows_any_missing:,} ({pct_any_missing:.1f}%)  "
    f"(manuscript states 36,704 / 53.3%)"
)