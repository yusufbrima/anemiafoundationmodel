import pandas as pd
import numpy as np

from config import RESULTS_DIR, VARIABLES, CAL_COLS
from util import load_dhs_kids_data, preprocess_dhs

TARGET = VARIABLES[-1]
FEATURES = VARIABLES[:-1]

df = pd.read_csv("./dhs_country_file_summary.csv")

summary = []

# -----------------------------
# Loop over countries
# -----------------------------
for index, row in df.iterrows():

    country = row["country"]
    print(f"Processing: {country}")

    # -----------------------------
    # Load raw DHS data
    # -----------------------------
    data_dict = load_dhs_kids_data(df, indx=index, variables=VARIABLES)
    raw_df = data_dict["kr"]

    raw_total = len(raw_df)

    # -----------------------------
    # Safe sex handling (DHS: 1=male, 2=female)
    # -----------------------------
    sex_raw = raw_df["b4"]

    raw_male = (raw_df["b4"] == "male").sum()
    raw_female = (raw_df["b4"] == "female").sum()

    # -----------------------------
    # Preprocess
    # -----------------------------
    clean_df = preprocess_dhs(raw_df, FEATURES, CAL_COLS, TARGET)

    proc_total = len(clean_df)

    sex_proc = clean_df["b4"]
    # print(raw_df["b4"].unique())
    # break
    proc_male = (sex_proc == "male").sum()
    proc_female = (sex_proc == "female").sum()


    # -----------------------------
    # Derived metrics (VERY IMPORTANT for paper)
    # -----------------------------
    drop_pct = 100 * (raw_total - proc_total) / raw_total if raw_total > 0 else np.nan

    female_ratio_raw = raw_female / raw_total if raw_total > 0 else np.nan
    female_ratio_proc = proc_female / proc_total if proc_total > 0 else np.nan

    sex_ratio_proc = proc_female / proc_male if proc_male > 0 else np.nan


    # -----------------------------
    # Optional but HIGHLY recommended stats
    # -----------------------------
    mean_age = clean_df["hw1"].mean()

    missing_rate = clean_df.isna().mean().mean()

    # -----------------------------
    # Store results
    # -----------------------------
    summary.append({
        "country": country,

        "raw_total": raw_total,
        "processed_total": proc_total,

        "raw_male_total": raw_male,
        "raw_female_total": raw_female,

        "processed_male": proc_male,
        "processed_female": proc_female,

        "female_ratio_raw": female_ratio_raw,
        "female_ratio_processed": female_ratio_proc,
        "sex_ratio_processed": sex_ratio_proc,

        "drop_pct": drop_pct,
        "missing_rate": missing_rate,

        "mean_age_months": mean_age
    })

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        f"{RESULTS_DIR}/dhs_country_data_summary.csv",
        index=False
    )
    # break
print("\nSaved: dhs_country_data_summary.csv")
print(summary_df.head())