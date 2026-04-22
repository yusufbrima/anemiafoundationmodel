import pandas as pd
from pathlib import Path

# ==========================
# Paths
# ==========================
ROOT_DIR = Path("/Users/ybrima/DHS_Databases")
SUMMARY_PATH = "./dhs_country_file_summary.csv"
OUTPUT_DIR = ROOT_DIR / "harmonized_output" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Load registry
# ==========================
df_registry = pd.read_csv(SUMMARY_PATH)

# ==========================
# Helper: safe loader
# ==========================
def load_dta(path):
    try:
        return pd.read_stata(path, convert_categoricals=True)
    except Exception as e:
        print(f"⚠️ Failed loading {path}: {e}")
        return None

# ==========================
# Helper: merge safely
# ==========================
def safe_merge(left, right, keys, how="inner"):
    if left is None or right is None:
        return left
    common_keys = [k for k in keys if k in left.columns and k in right.columns]
    if len(common_keys) == 0:
        return left
    return pd.merge(left, right, on=common_keys, how=how)

# ==========================
# Main loop
# ==========================
for country in df_registry["country"].unique():

    print(f"\n==============================")
    print(f"Processing country: {country}")
    print(f"==============================")

    row = df_registry[df_registry["country"] == country].iloc[0]

    # --------------------------
    # Load IR (anchor dataset)
    # --------------------------
    ir_files = row["ir_files"]
    df_ir = None

    if pd.notna(ir_files):
        ir_paths = ir_files.split("; ")
        dfs = [load_dta(p) for p in ir_paths]
        dfs = [d for d in dfs if d is not None]
        if dfs:
            df_ir = pd.concat(dfs, ignore_index=True)
            print(f"IR loaded: {df_ir.shape}")

    # --------------------------
    # Load OB
    # --------------------------
    ob_files = row["ob_files"]
    df_ob = None

    if pd.notna(ob_files):
        ob_paths = ob_files.split("; ")
        dfs = [load_dta(p) for p in ob_paths]
        dfs = [d for d in dfs if d is not None]
        if dfs:
            df_ob = pd.concat(dfs, ignore_index=True)
            print(f"OB loaded: {df_ob.shape}")

    # --------------------------
    # Load AR
    # --------------------------
    ar_files = row["ar_files"]
    df_ar = None

    if pd.notna(ar_files):
        ar_paths = ar_files.split("; ")
        dfs = [load_dta(p) for p in ar_paths]
        dfs = [d for d in dfs if d is not None]
        if dfs:
            df_ar = pd.concat(dfs, ignore_index=True)
            print(f"AR loaded: {df_ar.shape}")

    # --------------------------
    # Load HR
    # --------------------------
    hr_files = row["hr_files"]
    df_hr = None

    if pd.notna(hr_files):
        hr_paths = hr_files.split("; ")
        dfs = [load_dta(p) for p in hr_paths]
        dfs = [d for d in dfs if d is not None]
        if dfs:
            df_hr = pd.concat(dfs, ignore_index=True)
            print(f"HR loaded: {df_hr.shape}")

    # ==========================
    # MERGING LOGIC
    # ==========================

    df = df_ir

    # IR ↔ OB (biomarkers)
    if df is not None and df_ob is not None:
        print("Merging IR ↔ OB")
        df = safe_merge(df, df_ob, keys=[
            "caseid", "v001", "v002", "v003",
            "cluster", "hhid", "line"
        ])

    # IR ↔ AR (HIV datasets often separate)
    if df is not None and df_ar is not None:
        print("Merging IR ↔ AR")
        df = safe_merge(df, df_ar, keys=[
            "caseid", "v001", "v002", "v003"
        ])

    # IR ↔ HR (household context)
    if df is not None and df_hr is not None:
        print("Merging IR ↔ HR")
        df = safe_merge(df, df_hr, keys=[
            "v001", "v002"
        ])

    if df is None:
        print("⚠️ No IR base dataset found, skipping country")
        continue

    # ==========================
    # SAVE
    # ==========================
    out_path = OUTPUT_DIR / f"{country}_harmonized.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Saved: {out_path} | Shape: {df.shape}")