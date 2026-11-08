import pandas as pd
import numpy as np

from config import RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs, prepare_ml_data

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
# MAIN LOOP — COUNT ONLY, NO MODEL TRAINING
# =========================================================
count_rows = []

for index, row in df.iterrows():

    country = row["country"]
    print(f"\nProcessing: {country}")

    data_dict = load_dhs_kids_data(df, indx=index, variables=VARIABLES)
    ik_df = data_dict["kr"]

    df_clean = preprocess_dhs(ik_df, FEATURES, CAL_COLS, TARGET)
    if "hw1" in df_clean.columns:
        df_clean["age_group"] = pd.cut(
            pd.to_numeric(df_clean["hw1"], errors="coerce"),
            bins=[6, 12, 24, 36, 48, 60],
            labels=["6–11m", "12–23m", "24–35m", "36–47m", "48–59m"],
            right=False
        )
    # if "hw1" in df_clean.columns:
    #     df_clean["age_group"] = pd.cut(
    #         df_clean["hw1"],
    #         bins=[0, 6, 12, 24, 36, 48, 60],
    #         labels=["0–5m", "6–11m", "12–23m", "24–35m", "36–47m", "48–59m"]
    #     )

    for col in ["b4", "v190", "v106", "v025"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)

    # =====================================================
    # SUBGROUP LOOP
    # =====================================================
    for subgroup_name, subgroup_col in SUBGROUPS.items():

        # Column missing entirely from the dataset
        if subgroup_col not in df_clean.columns:
            count_rows.append({
                "country": country,
                "subgroup": subgroup_name,
                "subgroup_col": subgroup_col,
                "value": "ALL",
                "n_raw": 0,
                "n_after_cleaning": 0,
                "n_class_0": np.nan,
                "n_class_1": np.nan,
                "ml_valid": False,
                "skip_reason": "column_missing"
            })
            continue

        values = df_clean[subgroup_col].dropna().unique()

        for val in values:

            sub_df = df_clean[df_clean[subgroup_col] == val].copy()
            n_raw = len(sub_df)

            sub_df = sub_df.replace([np.inf, -np.inf], np.nan)
            sub_df = sub_df.dropna(subset=[TARGET] + FEATURES + CAL_COLS)
            n_clean = len(sub_df)

            # Count class distribution after cleaning
            if TARGET in sub_df.columns and n_clean > 0:
                class_counts = sub_df[TARGET].value_counts()
                n_class_0 = int(class_counts.get(0, 0))
                n_class_1 = int(class_counts.get(1, 0))
            else:
                n_class_0 = np.nan
                n_class_1 = np.nan

            # --- Determine skip reason ---
            if n_clean < 50:
                skip_reason = f"insufficient_n_after_cleaning (n={n_clean})"
                ml_valid = False
            else:
                # Check whether train split would have both classes
                try:
                    _, _, y_train, _, _ = prepare_ml_data(
                        sub_df, FEATURES, CAL_COLS, TARGET
                    )
                    if len(np.unique(y_train)) < 2:
                        skip_reason = "single_class_in_train_split"
                        ml_valid = False
                    else:
                        skip_reason = None
                        ml_valid = True
                except Exception as e:
                    skip_reason = f"prepare_ml_data_error: {str(e)}"
                    ml_valid = False

            count_rows.append({
                "country": country,
                "subgroup": subgroup_name,
                "subgroup_col": subgroup_col,
                "value": str(val),
                "n_raw": n_raw,
                "n_after_cleaning": n_clean,
                "n_class_0": n_class_0,
                "n_class_1": n_class_1,
                "ml_valid": ml_valid,
                "skip_reason": skip_reason if not ml_valid else "—"
            })

# =========================================================
# SAVE
# =========================================================
counts_df = pd.DataFrame(count_rows)
out_path = f"{RESULTS_DIR}/supplementary_subgroup_table.csv"
counts_df.to_csv(out_path, index=False)

print(f"\nSaved subgroup counts to: {out_path}")
print(f"Total rows: {len(counts_df)}")
print(f"  ML-valid subgroups : {counts_df['ml_valid'].sum()}")
print(f"  Skipped subgroups  : {(~counts_df['ml_valid']).sum()}")
print("\nSkip reason breakdown:")
print(counts_df[~counts_df["ml_valid"]]["skip_reason"].value_counts())
print("\nSample output:")
print(counts_df.head(10).to_string(index=False))