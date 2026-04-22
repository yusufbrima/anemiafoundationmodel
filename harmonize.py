import pandas as pd
from pathlib import Path

# ==========================
# Paths
# ==========================
ROOT_DIR = Path("/content/drive/MyDrive/ResearchProjects/DHS/DHS_Databases")
# OUTPUT_DIR = ROOT_DIR / "harmonized_output"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Helper: case-insensitive file finder
# ==========================
def find_files(folder, keyword):
    """
    Finds all .dta files containing keyword (case-insensitive)
    """
    return sorted([
        str(f) for f in folder.rglob("*")
        if keyword in f.name.upper() and f.suffix.lower() == ".dta"
    ])
    # return sorted([
    #     f.name for f in folder.rglob("*")
    #     if keyword in f.name.upper() and f.suffix.lower() == ".dta"
    # ])

# ==========================
# Main loop
# ==========================
rows = []

for country_folder in ROOT_DIR.iterdir():

    if country_folder.is_dir():

        country = country_folder.name.replace("_unzipped", "")
        print(f"\nProcessing: {country}")
        # Individual Recode
        ir_files = find_files(country_folder, "IR")
        # Other Biomarkers Datasets
        ob_files = find_files(country_folder, "OB")
        # HIV Test Results Recode
        ar_files = find_files(country_folder, "AR")
        # Household Recode
        hr_files = find_files(country_folder, "HR")

        # Births Recode
        br_files = find_files(country_folder, "BR")

        # Men's Recode
        mr_files = find_files(country_folder, "MR")

        # Household Member Recode
        pr_files = find_files(country_folder, "PR")

        # Children Recode
        kr_files = find_files(country_folder, "KR")

        rows.append({
            "country": country,
            "ir_files": "; ".join(ir_files) if ir_files else None,
            "ob_files": "; ".join(ob_files) if ob_files else None,
            "ar_files": "; ".join(ar_files) if ar_files else None,
            "hr_files": "; ".join(hr_files) if hr_files else None,
            "br_files": "; ".join(br_files) if br_files else None,
            "mr_files": "; ".join(mr_files) if mr_files else None,
            "pr_files": "; ".join(pr_files) if pr_files else None,
            "kr_files": "; ".join(kr_files) if kr_files else None,
        })

# ==========================
# Save summary table
# ==========================
df_summary = pd.DataFrame(rows)

output_file = "./dhs_country_file_summary.csv"
df_summary.to_csv(output_file, index=False)

print(f"\n✅ Saved clean registry to: {output_file}")